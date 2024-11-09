//! Process management syscalls

use crate::{
    config::MAX_SYSCALL_NUM , mm::{map_area_from_token, map_user_ptr_to_kernel, unmap_area_from_token, MapPermission, VirtAddr }, task::{
        change_program_brk, exit_current_and_run_next, get_current_task_exec_time, get_current_task_status, get_current_task_syscall_times, suspend_current_and_run_next, TaskStatus
    }, timer::{get_time_ms, get_time_us}
};

#[repr(C)]
#[derive(Debug)]
pub struct TimeVal {
    pub sec: usize,
    pub usec: usize,
}

/// Task information
#[allow(dead_code)]
pub struct TaskInfo {
    /// Task status in it's life cycle
    status: TaskStatus,
    /// The numbers of syscall called by task
    syscall_times: [u32; MAX_SYSCALL_NUM],
    /// Total running time of task
    time: usize,
}

/// task exits and submit an exit code
pub fn sys_exit(_exit_code: i32) -> ! {
    trace!("kernel: sys_exit");
    exit_current_and_run_next();
    panic!("Unreachable in sys_exit!");
}

/// current task gives up resources for other tasks
pub fn sys_yield() -> isize {
    trace!("kernel: sys_yield");
    suspend_current_and_run_next();
    0
}

/// YOUR JOB: get time with second and microsecond
/// HINT: You might reimplement it with virtual memory management.
/// HINT: What if [`TimeVal`] is splitted by two pages ?
pub fn sys_get_time(_ts: *mut TimeVal, _tz: usize) -> isize {
    trace!("kernel: sys_get_time");

    let time_us = get_time_us();
    let result = map_user_ptr_to_kernel(_ts);

    if result.is_err() {
        return -1;
    }

    let virt_ts = result.unwrap();

    unsafe{
        (*virt_ts).sec = time_us / 1_000_000;
        (*virt_ts).usec = time_us % 1_000_000;
    }
    0
}

/// YOUR JOB: Finish sys_task_info to pass testcases
/// HINT: You might reimplement it with virtual memory management.
/// HINT: What if [`TaskInfo`] is splitted by two pages ?
pub fn sys_task_info(_ti: *mut TaskInfo) -> isize {
    trace!("kernel: sys_task_info");
    let status = get_current_task_status();
    let syscall_times = get_current_task_syscall_times();
    let time = get_time_ms() - get_current_task_exec_time();

    let result = map_user_ptr_to_kernel(_ti);

    if result.is_err() {
        return -1;
    }

    let virt_ti = result.unwrap();

    unsafe{
        (*virt_ti).status = status;
        (*virt_ti).syscall_times = syscall_times;
        (*virt_ti).time = time;
    }
    0
}

// YOUR JOB: Implement mmap.
pub fn sys_mmap(_start: usize, _len: usize, _port: usize) -> isize {
    trace!("kernel: sys_mmap");

    if _port > 7 || _port == 0 {
        return -1
    }

    if VirtAddr(_start).page_offset() != 0 {
        return -1
    }

    let mut permission: MapPermission = MapPermission::U;

    if _port & 1 != 0 {
        permission |= MapPermission::R
    }

    if _port & 2 != 0 {
        permission |= MapPermission::W
    }

    if _port & 4 != 0 {
        permission |= MapPermission::X

    }

    let result = map_area_from_token(_start, _len, permission);
    if result.is_ok() {
        0
    } else {
        -1
    }
}

// YOUR JOB: Implement munmap.
pub fn sys_munmap(_start: usize, _len: usize) -> isize {
    trace!("kernel: sys_munmap");

    if VirtAddr(_start).page_offset() != 0 {
        return -1
    }
    
    let result = unmap_area_from_token(_start, _len);

    if result.is_ok() {
        0
    } else {
        -1
    }
}
/// change data segment size
pub fn sys_sbrk(size: i32) -> isize {
    trace!("kernel: sys_sbrk");
    if let Some(old_brk) = change_program_brk(size) {
        old_brk as isize
    } else {
        -1
    }
}
