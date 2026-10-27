"""Persistent desktop SSH connection with caller-supplied authentication dialogs."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import socket
import shlex
import time


class SubmissionProgress:
    """Retain raw diagnostics; omit only the known no-terminal mesg notice."""

    def __init__(self, output, log):
        self.output, self.log = output, log
        self.pending = ''

    def __call__(self, text):
        self.log.write(text)
        self.log.flush()
        self.pending += text
        while '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            self.display(line + '\n')

    def display(self, line):
        if line.rstrip('\r\n') != 'mesg: ttyname failed: Inappropriate ioctl for device':
            self.output(line)

    def finish(self):
        if self.pending:
            self.display(self.pending)
            self.pending = ''


def submission_shell_command(command):
    """Load the login environment before running the already-quoted launcher."""
    body = (
        'if ! command -v sbatch >/dev/null 2>&1; then '
        'if [ -x /usr/local/slurm/bin/sbatch ]; then '
        'export PATH="$PATH:/usr/local/slurm/bin"; fi; fi; '
        'command -v sbatch >/dev/null 2>&1 || { '
        'echo "HPC sbatch is unavailable in the login environment. No jobs submitted." >&2; '
        'exit 127; }; exec ' + command
    )
    return 'bash -lc ' + shlex.quote(body)


class DesktopSSH:
    def __init__(self, known_hosts):
        self.known_hosts = Path(known_hosts)
        self.transport = None
        self.host = self.username = ''

    @property
    def connected(self):
        return bool(self.transport and self.transport.is_active() and self.transport.is_authenticated())

    def close(self):
        if self.transport:
            self.transport.close()
        self.transport = None

    def connect(self, host, username, prompt):
        """prompt(kind, data) runs the GUI dialog; passwords never enter saved state."""
        import paramiko
        from .desktop_jobs import ssh_arguments
        ssh_arguments(host, username)  # Validate destination without starting a process.
        self.close()
        known = paramiko.HostKeys()
        for path in (Path.home() / '.ssh' / 'known_hosts', self.known_hosts):
            if path.exists():
                known.load(str(path))
        expected = known.lookup(host)
        sock = socket.create_connection((host, 22), timeout=20)
        transport = paramiko.Transport(sock)
        self.transport = transport
        transport.auth_timeout = 240
        transport.banner_timeout = 30
        try:
            if expected:
                options = transport.get_security_options()
                preferred = tuple(k for k in options.key_types if k in expected)
                options.key_types = preferred + tuple(k for k in options.key_types if k not in preferred)
            transport.start_client(timeout=30)
            key = transport.get_remote_server_key()
            if expected:
                if not known.check(host, key):
                    raise ValueError('The SSH host key differs from the saved key. Connection stopped before authentication.')
            else:
                fingerprint = 'SHA256:' + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
                if not prompt('host_key', {'host': host, 'fingerprint': fingerprint, 'type': key.get_name()}):
                    raise ValueError('Connection cancelled.')
                saved = paramiko.HostKeys()
                if self.known_hosts.exists(): saved.load(str(self.known_hosts))
                saved.add(host, key.get_name(), key)
                self.known_hosts.parent.mkdir(parents=True, exist_ok=True)
                saved.save(str(self.known_hosts))
            try:
                allowed = transport.auth_none(username)
            except paramiko.BadAuthenticationType as error:
                allowed = error.allowed_types
            if not transport.is_authenticated() and 'password' in allowed:
                response = prompt('authentication', {'title': 'HPC login',
                    'instructions': f'Password for {username}@{host}', 'prompts': [('Password', False)]})
                if response is None: raise ValueError('Connection cancelled.')
                password = response.pop(0)
                try:
                    allowed = transport.auth_password(username, password, fallback=False)
                except paramiko.BadAuthenticationType as error:
                    allowed = error.allowed_types
                finally:
                    password = None
            if not transport.is_authenticated() and 'keyboard-interactive' in allowed:
                def handler(title, instructions, prompts):
                    answers = prompt('authentication', {'title': title or 'HPC verification',
                        'instructions': instructions, 'prompts': prompts})
                    if answers is None: raise ValueError('Connection cancelled.')
                    return answers
                transport.auth_interactive(username, handler)
            if not transport.is_authenticated():
                raise ValueError('The server did not accept password/MFA authentication. Reconnect to try again.')
            transport.set_keepalive(30)
            self.host, self.username = host, username
            return f'Connected: {username}@{host}'
        except Exception:
            self.close()
            raise

    def query(self, command, timeout=45):
        """Bounded read-only scheduler query; no submission preflight or retries."""
        if not self.connected:
            raise ConnectionError('Connect to HPC to refresh live status.')
        channel = self.transport.open_session(timeout=20)
        chunks = []
        try:
            channel.set_combine_stderr(True)
            body = 'export PATH="$PATH:/usr/local/slurm/bin"; ' + command
            channel.exec_command('bash -lc ' + shlex.quote(body))
            deadline = time.monotonic() + timeout
            while True:
                while channel.recv_ready():
                    chunks.append(channel.recv(32768).decode('utf-8', errors='replace'))
                if channel.exit_status_ready():
                    while channel.recv_ready():
                        chunks.append(channel.recv(32768).decode('utf-8', errors='replace'))
                    code = channel.recv_exit_status()
                    if code:
                        raise RuntimeError('Status query failed; previous display is stale.\n' + ''.join(chunks)[-2000:])
                    return ''.join(chunks)
                if time.monotonic() >= deadline:
                    raise TimeoutError('Status query timed out; previous display is stale.')
                time.sleep(.1)
        finally:
            channel.close()

    def execute(self, command, output):
        """Execute once on the existing session; never reconnect/retry submission."""
        if not self.connected:
            raise ConnectionError('Connection lost. Connect again before submitting.')
        channel = self.transport.open_session(timeout=20)
        try:
            channel.set_combine_stderr(True)
            channel.exec_command(submission_shell_command(command))
            while True:
                while channel.recv_ready():
                    output(channel.recv(32768).decode('utf-8', errors='replace'))
                if channel.exit_status_ready():
                    while channel.recv_ready():
                        output(channel.recv(32768).decode('utf-8', errors='replace'))
                    code = channel.recv_exit_status()
                    if code != 0:
                        raise RuntimeError(f'Submission exited with code {code}. Review the output and accepted job IDs before retrying.')
                    return code
                if not self.connected:
                    raise ConnectionError('Connection interrupted. Some jobs may already be accepted; read job IDs before any retry.')
                time.sleep(.1)
        finally:
            channel.close()
