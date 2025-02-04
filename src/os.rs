use crate::types::{mi_assert, mi_assert_internal};
use crate::types::{MiOsTld, MI_SEGMENT_SIZE};
use libc::{
    madvise, mmap, munmap, MADV_DONTNEED, MADV_FREE, MAP_FAILED, MAP_PRIVATE, PROT_READ, PROT_WRITE,
};
use libc::{MAP_ANONYMOUS, MAP_FIXED};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Once;
use std::{io, ptr};

pub(crate) fn mi_align_up(sz: usize, alignment: usize) -> usize {
    let x = (sz / alignment) * alignment;
    if x < sz {
        if let Some(new_x) = x.checked_add(alignment) {
            if new_x >= sz {
                return new_x;
            }
        }
        return 0; // overflow
    }
    x
}

pub(crate) fn mi_align_up_ptr(p: *mut u8, alignment: usize) -> *mut u8 {
    mi_align_up(p as usize, alignment) as *mut u8
}

pub(crate) fn mi_align_down(sz: usize, alignment: usize) -> usize {
    (sz / alignment) * alignment
}

pub(crate) fn mi_align_down_ptr(p: *mut u8, alignment: usize) -> *mut u8 {
    mi_align_down(p as usize, alignment) as *mut u8
}

// Cached OS page size
pub(crate) fn mi_os_page_size() -> usize {
    static PAGE_SIZE: AtomicUsize = AtomicUsize::new(0);
    static INIT: Once = Once::new();

    INIT.call_once(|| {
        let actual_size = {
            let result = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if result > 0 {
                result as usize
            } else {
                4096 // 默认页大小
            }
        };

        PAGE_SIZE.store(actual_size, Ordering::Relaxed);
    });

    PAGE_SIZE.load(Ordering::Relaxed)
}

fn mi_munmap(addr: *mut u8, size: usize) -> bool {
    if addr.is_null() || size == 0 {
        return true;
    }
    let err = unsafe { munmap(addr as *mut _, size) == -1 };

    if err {
        eprintln!(
            "munmap failed: {}, addr: {:p}, size: {}",
            io::Error::last_os_error(),
            addr,
            size
        );
        false
    } else {
        true
    }
}

fn mi_mmap(addr: Option<*mut u8>, size: usize, extra_flags: i32) -> *mut u8 {
    if size == 0 {
        return null_mut();
    }
    let mut p: *mut u8;

    unsafe {
        let mut flags = MAP_PRIVATE | MAP_ANONYMOUS | extra_flags;

        if addr.is_some() {
            flags |= MAP_FIXED;
            // | if cfg!(any(MAP_FIXED_NOREPLACE, MAP_EXCL)) {
            //     MAP_FIXED
            // } else {
            //     0
            // };
        }

        p = {
            let prot_flags = PROT_READ | PROT_WRITE;
            let p = mmap(
                addr.unwrap_or(ptr::null_mut()) as *mut _,
                size,
                prot_flags,
                flags,
                -1,
                0,
            );

            if p == MAP_FAILED {
                ptr::null_mut()
            } else {
                p as *mut u8
            }
        };
        if let Some(addr) = addr {
            if p != addr {
                mi_munmap(p, size);
                p = ptr::null_mut();
            }
        }
    }

    mi_assert(p.is_null() || addr.is_none() || (addr.is_some() && p == addr.unwrap()));

    p
}

fn mi_os_page_align_region(
    addr: *mut u8,
    size: usize,
    mut newsize: Option<&mut usize>,
) -> Option<*mut u8> {
    mi_assert(!addr.is_null() && size > 0);
    if let Some(ref mut newsize) = newsize {
        **newsize = 0;
    }
    if size == 0 || addr.is_null() {
        return None;
    }

    // page align conservatively within the range
    let start = mi_align_up_ptr(addr, mi_os_page_size());
    let end = mi_align_down_ptr(unsafe { addr.add(size) }, mi_os_page_size());
    let diff = (end as isize - start as isize) as isize;
    if diff <= 0 {
        return None;
    }

    mi_assert_internal(diff as usize <= size);

    if let Some(ref mut newsize) = newsize {
        **newsize = diff as usize;
    }
    Some(start)
}

pub(crate) fn mi_os_reset(addr: *mut u8, size: usize) -> bool {
    let mut csize = 0;
    let start = match mi_os_page_align_region(addr, size, Some(&mut csize)) {
        Some(s) => s,
        None => return true,
    };

    let mut advice = MADV_FREE;
    unsafe {
        let mut err = madvise(start as *mut _, csize, advice);
        if err != 0 && *libc::__errno_location() == libc::EINVAL && advice == MADV_FREE {
            advice = MADV_DONTNEED;
            err = madvise(start as *mut _, csize, advice);
        }
        if err != 0 {
            eprintln!(
                "madvise reset error: start: {:p}, csize: {}, errno: {}",
                start,
                csize,
                io::Error::last_os_error()
            );
        }
        err == 0
    }
}

pub(crate) fn mi_os_shrink(p: *mut u8, oldsize: usize, newsize: usize) -> bool {
    mi_assert_internal(oldsize > newsize && !p.is_null());
    if oldsize < newsize || p.is_null() {
        return false;
    }
    if oldsize == newsize {
        return true;
    }

    let addr = unsafe { p.add(newsize) };
    let mut size = 0;
    let start = {
        let start = mi_os_page_align_region(addr, oldsize - newsize, Some(&mut size));
        if start != Some(addr) {
            return false;
        }
        start.unwrap()
    };

    mi_munmap(start, size)
}

/* -----------------------------------------------------------
  OS allocation using mmap/munmap
----------------------------------------------------------- */

pub(crate) fn mi_os_alloc(size: usize) -> *mut u8 {
    if size == 0 {
        return null_mut();
    }
    let p = mi_mmap(None, size, 0);
    p
}

pub(crate) fn mi_os_free(p: *mut u8, size: usize) {
    mi_munmap(p, size);
}

// Slow but guaranteed way to allocated aligned memory
// by over-allocating and then reallocating at a fixed aligned
// address that should be available then.
fn mi_os_alloc_aligned_ensured(size: usize, alignment: usize, trie: usize) -> *mut u8 {
    if trie >= 3 {
        return null_mut(); // stop recursion (only on Windows)
    }
    let alloc_size = size + alignment;
    mi_assert(alloc_size >= size); // overflow?
    if alloc_size < size {
        return null_mut();
    }

    // allocate a chunk that includes the alignment
    let p = mi_mmap(None, alloc_size, 0);
    if p.is_null() {
        return null_mut();
    }

    // create an aligned pointer in the allocated area
    let aligned_p = mi_align_up_ptr(p, alignment);
    mi_assert(!aligned_p.is_null());

    // we selectively unmap parts around the over-allocated area.
    let pre_size = (aligned_p as usize) - (p as usize);
    let mid_size = mi_align_up(size, mi_os_page_size());
    let post_size = alloc_size - pre_size - mid_size;
    if pre_size > 0 {
        mi_munmap(p, pre_size);
    }
    if post_size > 0 {
        mi_munmap((aligned_p as usize + mid_size) as *mut _, post_size);
    }

    mi_assert((aligned_p as usize) % alignment == 0);
    aligned_p
}

// Allocate an aligned block.
// Since `mi_mmap` is relatively slow we try to allocate directly at first and
// hope to get an aligned address; only when that fails we fall back
// to a guaranteed method by overallocating at first and adjusting.
// TODO: use VirtualAlloc2 with alignment on Windows 10 / Windows Server 2016.
pub(crate) unsafe fn mi_os_alloc_aligned(
    size: usize,
    alignment: usize,
    tld: *mut MiOsTld,
) -> *mut u8 {
    if size == 0 {
        return null_mut();
    }
    if alignment < 1024 {
        return mi_os_alloc(size);
    }

    let mut p = os_pool_alloc(size, alignment, &mut *tld);
    if !p.is_null() {
        return p;
    }

    let suggest = None;

    // on BSD, use the aligned mmap api
    #[cfg(any(target_os = "netbsd", target_os = "freebsd"))]
    {
        let n = alignment.trailing_zeros();
        if (1 << n) == alignment && n >= 12 {
            // alignment is a power of 2 and >= 4096
            p = mi_mmap(suggest, size, 0); // use the NetBSD/freeBSD aligned flags
        }
    }

    if p.is_null() && ((*tld).mmap_next_probable % alignment) == 0 {
        // if the next probable address is aligned,
        // then try to just allocate `size` and hope it is aligned...
        p = mi_mmap(suggest, size, 0);
        if p.is_null() {
            return null_mut();
        }
    }

    if p.is_null() || (p as usize) % alignment != 0 {
        // if `p` is not yet aligned after all, free the block and use a slower
        // but guaranteed way to allocate an aligned block
        if !p.is_null() {
            mi_munmap(p, size);
        }
        p = mi_os_alloc_aligned_ensured(size, alignment, 0);
    }
    if !p.is_null() {
        // next probable address is the page-aligned address just after the newly allocated area.
        let alloc_align = mi_os_page_size(); // page size on other OS's

        let probable_size = MI_SEGMENT_SIZE;
        if (*tld).mmap_previous > p {
            // Linux tends to allocate downward
            (*tld).mmap_next_probable = mi_align_down((p as usize) - probable_size, alloc_align);
        } else {
            // Otherwise, guess the next address is page aligned `size` from current pointer
            (*tld).mmap_next_probable = mi_align_up((p as usize) + probable_size, alloc_align);
        }
        (*tld).mmap_previous = p;
    }

    p
}

// Pooled allocation: on 64-bit systems with plenty
// of virtual addresses, we allocate 10 segments at the
// time to minimize `mmap` calls and increase aligned
// allocations. This is only good on systems that
// do overcommit so we put it behind the `MIMALLOC_POOL_COMMIT` option.
// For now, we disable it on windows as VirtualFree must
// be called on the original allocation and cannot be called
// for individual fragments.

const MI_POOL_ALIGNMENT: usize = MI_SEGMENT_SIZE;
const MI_POOL_SIZE: usize = 10 * MI_POOL_ALIGNMENT;

pub(crate) unsafe fn os_pool_alloc(size: usize, alignment: usize, tld: &mut MiOsTld) -> *mut u8 {
    // #if defined(_WIN32) || (MI_INTPTR_SIZE<8)
    // return null_mut();
    if !false {
        return null_mut();
    }
    if alignment != MI_POOL_ALIGNMENT {
        return null_mut();
    }
    let size = mi_align_up(size, MI_POOL_ALIGNMENT);
    if size == 0 || size > MI_POOL_SIZE {
        return null_mut();
    }
    let size = size;

    if tld.pool_available == 0 {
        tld.pool = mi_os_alloc_aligned_ensured(MI_POOL_SIZE, MI_POOL_ALIGNMENT, 0);
        if tld.pool.is_null() {
            return null_mut();
        }
        tld.pool_available += MI_POOL_SIZE;
    }

    if size > tld.pool_available {
        return null_mut();
    }
    let p = tld.pool;
    tld.pool_available -= size;
    tld.pool = unsafe { tld.pool.add(size) };
    p
}
