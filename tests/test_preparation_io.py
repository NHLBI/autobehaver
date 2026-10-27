import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from autobehaver.preparation_io import io_results, preparation_progress


def test_bounded_parallel_order_and_progress():
    active = peak = 0
    lock=threading.Lock(); messages=[]
    def work(i):
        nonlocal active,peak
        with lock: active+=1;peak=max(peak,active)
        time.sleep(.02*(4-i%4))
        with lock:active-=1
        return i*i
    with preparation_progress(messages.append):
        actual=list(io_results(work,range(24),label='Example',workers=4))
    assert actual==[i*i for i in range(24)]
    assert 1<peak<=4
    assert '24 / 24' in messages[-2]
    assert 'complete' in messages[-1]


def test_error_is_not_success():
    messages=[]
    def work(i):
        if i==2:raise ValueError('bad input')
        return i
    with preparation_progress(messages.append),pytest.raises(ValueError,match='bad input'):
        list(io_results(work,range(5),label='Check'))
    assert not any('complete' in m for m in messages)


def test_task_callbacks_are_isolated():
    def task(name):
        messages=[]
        with preparation_progress(messages.append):
            assert list(io_results(abs,[-1,-2],label=name))==[1,2]
        return messages
    with ThreadPoolExecutor(max_workers=2) as pool:
        a,b=list(pool.map(task,['A','B']))
    assert all(m.startswith('A:') for m in a)
    assert all(m.startswith('B:') for m in b)
