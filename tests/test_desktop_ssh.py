"""Exercise password/MFA and connection reuse against a local in-process SSH server."""
import socket
import threading
import time
import shlex
import subprocess
from pathlib import Path

import pytest

paramiko = pytest.importorskip('paramiko')
from autobehaver.desktop_ssh import DesktopSSH, SubmissionProgress, submission_shell_command


def test_submission_progress_retains_errors_and_raw_split_output():
    import io
    warning = 'mesg: ttyname failed: Inappropriate ioctl for device'
    raw = warning + '\r\naccepted 123\nsbatch: error: failed\nother: ' + warning + '\nlast partial line'
    # Network chunks may divide a warning at any byte; only its exact line is hidden.
    for split in range(len(raw) + 1):
        output, log = [], io.StringIO()
        progress = SubmissionProgress(output.append, log)
        progress(raw[:split]); progress(raw[split:]); progress.finish()
        assert log.getvalue() == raw
        assert ''.join(output) == raw[len(warning) + 2:]
    output, log = [], io.StringIO()
    progress = SubmissionProgress(output.append, log)
    progress(warning); progress.finish()
    assert output == [] and log.getvalue() == warning


@pytest.fixture
def server(monkeypatch):
    listener = socket.socket(); listener.bind(('127.0.0.1', 0)); listener.listen(1)
    listener.settimeout(10)
    key = paramiko.RSAKey.generate(2048)
    state = {'mode': 'password', 'commands': [], 'auth_calls': 0}
    stop = threading.Event()
    class Server(paramiko.ServerInterface):
        def get_allowed_auths(self, username):
            return 'keyboard-interactive' if state['mode'] == 'mfa' else 'password'
        def check_auth_password(self, username, password):
            state['auth_calls'] += 1
            return paramiko.AUTH_SUCCESSFUL if username == 'user' and password == 'secret' else paramiko.AUTH_FAILED
        def check_auth_interactive(self, username, submethods):
            return paramiko.InteractiveQuery('Verification', 'Test MFA', ('Password', False), ('Code', False))
        def check_auth_interactive_response(self, responses):
            state['auth_calls'] += 1
            return paramiko.AUTH_SUCCESSFUL if responses == ['secret', '123456'] else paramiko.AUTH_FAILED
        def check_channel_request(self, kind, chanid):
            return paramiko.OPEN_SUCCEEDED
        def check_channel_exec_request(self, channel, command):
            state['commands'].append(command.decode())
            channel.test_command_received = True
            return True
    def serve():
        transport = None
        try:
            sock, _ = listener.accept()
            transport = paramiko.Transport(sock); transport.add_server_key(key)
            transport.start_server(server=Server())
            while not stop.is_set() and transport.is_active():
                channel = transport.accept(.1)
                if channel:
                    deadline = time.monotonic() + 5
                    while not channel.closed and time.monotonic() < deadline:
                        if getattr(channel, 'test_command_received', False):
                            # Let the exec request callback complete before writing.
                            time.sleep(.05); break
                        time.sleep(.01)
                    channel.send(b'accepted 123\n'); channel.send_exit_status(0); channel.close()
        except (EOFError, OSError, paramiko.SSHException):
            pass
        finally:
            if transport: transport.close()
    thread = threading.Thread(target=serve, daemon=True); thread.start()
    original = socket.create_connection
    monkeypatch.setattr(socket, 'create_connection', lambda address, timeout=None: original(listener.getsockname(), timeout=timeout))
    yield state, key
    stop.set(); listener.close(); thread.join(timeout=3)


def test_password_login_reuses_connection_for_two_commands(tmp_path, server):
    state, key = server
    ssh = DesktopSSH(tmp_path / 'known_hosts')
    prompts = []
    def prompt(kind, data):
        prompts.append(kind)
        return True if kind == 'host_key' else ['secret']
    ssh.connect('example.org', 'user', prompt)
    output = []
    assert ssh.connected
    ssh.execute('first', output.append); ssh.execute('second', output.append)
    assert state['commands'] == [submission_shell_command('first'), submission_shell_command('second')]
    assert state['auth_calls'] == 1
    assert ''.join(output) == 'accepted 123\naccepted 123\n'
    assert prompts == ['host_key', 'authentication']
    assert 'secret' not in (tmp_path / 'known_hosts').read_text()
    ssh.close(); assert not ssh.connected


def test_keyboard_interactive_mfa(tmp_path, server):
    state, key = server; state['mode'] = 'mfa'
    ssh = DesktopSSH(tmp_path / 'known_hosts')
    def prompt(kind, data):
        if kind == 'host_key': return True
        assert data['prompts'] == [('Password', False), ('Code', False)]
        return ['secret', '123456']
    ssh.connect('example.org', 'user', prompt)
    assert ssh.connected and state['auth_calls'] == 1
    ssh.close()


def test_changed_host_key_stops_before_password(tmp_path, server):
    state, key = server
    known = paramiko.HostKeys(); known.add('example.org', 'ssh-rsa', paramiko.RSAKey.generate(2048))
    path = tmp_path / 'known_hosts'; known.save(str(path))
    ssh = DesktopSSH(path)
    with pytest.raises(ValueError, match='differs'):
        ssh.connect('example.org', 'user', lambda *args: pytest.fail('Must not prompt for credentials'))
    assert state['auth_calls'] == 0 and not ssh.connected


def test_cancel_login_closes_connection(tmp_path, server):
    ssh = DesktopSSH(tmp_path / 'known_hosts')
    with pytest.raises(ValueError, match='cancelled'):
        ssh.connect('example.org', 'user', lambda kind, data: True if kind == 'host_key' else None)
    assert not ssh.connected


def test_submission_uses_login_shell_and_preserves_quoted_bundle():
    path = "/data/lab/job bundles/author's run/submit.sh"
    args = shlex.split(submission_shell_command('bash ' + shlex.quote(path)))
    assert args[:2] == ['bash', '-lc'] and len(args) == 3
    launcher = args[2].split('; exec ', 1)[1]
    assert shlex.split(launcher) == ['bash', path]


def test_missing_sbatch_fails_before_launcher_and_available_sbatch_runs_it(tmp_path):
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    if not bash.exists():
        import shutil
        binary = shutil.which('bash')
        if not binary: pytest.skip('Bash unavailable')
        bash = Path(binary)
    # Execute the actual preflight body in a controlled shell, with command-v
    # simulated so the test is independent of installed cluster tools.
    body = shlex.split(submission_shell_command('printf launcher_ran'))[2]
    missing = subprocess.run([str(bash), '--noprofile', '--norc', '-c',
        'command() { return 1; }; ' + body], capture_output=True, text=True)
    assert missing.returncode == 127
    assert 'launcher_ran' not in missing.stdout and 'No jobs submitted' in missing.stderr
    available = subprocess.run([str(bash), '--noprofile', '--norc', '-c',
        'command() { return 0; }; ' + body], capture_output=True, text=True)
    assert available.returncode == 0 and available.stdout == 'launcher_ran'


def test_status_query_uses_existing_connection_without_submission(tmp_path, server):
    state, key = server
    ssh = DesktopSSH(tmp_path / 'known_hosts')
    ssh.connect('example.org', 'user', lambda kind, data: True if kind == 'host_key' else ['secret'])
    assert ssh.query('sacct -n') == 'accepted 123\n'
    assert state['auth_calls'] == 1
    command = shlex.split(state['commands'][0])
    assert command[:2] == ['bash', '-lc']
    assert 'sacct -n' in command[2] and 'sbatch' not in command[2]
    ssh.close()
