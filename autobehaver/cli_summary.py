"""Presentation-only closing summaries; processing and exit status stay with callers."""
from contextvars import ContextVar
from functools import wraps
import sys
import time

_current = ContextVar('autobehaver_closing_summary', default=None)


def completion_details(**details):
    """Record already-known results; never inspect or modify scientific outputs."""
    state = _current.get()
    if state is not None:
        state.update(details)
        state['active'] = True


def validation_details(report, *, unit, output):
    counts = report.outcome_counts()
    # 'succeeded' can include pre-existing crops; do not label it newly processed.
    displayed = {'validated': counts.get('succeeded', 0),
                 'reused': counts.get('reused', 0), 'failed': counts.get('failed', 0)}
    for name, count in counts.items():
        if name not in ('succeeded', 'reused', 'failed') and count:
            displayed[name] = count
    status = {'failed': 'failed', 'warning': 'complete with warnings'}.get(report.status, 'complete')
    if status == 'complete' and not counts.get('succeeded', 0) and not counts.get('failed', 0):
        status = 'nothing to process'
    completion_details(status=status, unit=unit, counts=displayed, output=output)


def _emit(state, elapsed):
    if not state['active']:
        return
    seconds = max(0, int(elapsed))
    lines = [f"{state['stage']} - {state['status']}"]
    if state.get('counts'):
        lines.append(f"{state.get('unit', 'Items')}: " + ' | '.join(
            f'{count} {label}' for label, count in state['counts'].items()))
    lines.append(f'Elapsed: {seconds // 60}m {seconds % 60:02d}s')
    if state.get('output'):
        lines.append(f"Output location: {state['output']}")
    if state['status'] == 'failed':
        lines.append('See the error details; this stage did not complete successfully.')
    try:
        sys.stdout.flush()  # Keep the footer last when stdout/stderr share a log.
    except (OSError, ValueError):
        pass
    try:
        print('\n' + '\n'.join(lines), file=sys.stderr, flush=True)
    except (OSError, ValueError):
        pass  # A closed reporting stream must not change processing outcomes.


def reported_stage(stage, *, active=True, exit_code=False):
    """Print after the function and its finally blocks, preserving returns/errors."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            state = {'stage': stage, 'status': 'complete', 'active': active}
            token = _current.set(state)
            start = time.monotonic()
            try:
                result = function(*args, **kwargs)
                if exit_code and result not in (None, 0) and state['status'] != 'nothing to process':
                    state['status'] = 'failed'
                return result
            except KeyboardInterrupt:
                state['status'] = 'interrupted'
                raise
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    state['status'] = 'failed'
                raise
            except BaseException:
                state['status'] = 'failed'
                raise
            finally:
                try:
                    _emit(state, time.monotonic() - start)
                finally:
                    _current.reset(token)
        return wrapped
    return decorate
