/// The `MemorySet` struct represents an address space, which includes a page table and a collection of memory areas.
/// It provides methods to create new memory sets, manipulate memory areas, and manage the page table.
///
/// # Methods
///
/// - `new_bare() -> Self`
///   Creates a new empty `MemorySet`.
///
/// - `token(&self) -> usize`
///   Returns the page table token.
///
/// - `insert_framed_area(&mut self, start_va: VirtAddr, end_va: VirtAddr, permission: MapPermission)`
///   Inserts a framed area into the memory set.
///
/// - `push(&mut self, map_area: MapArea, data: Option<&[u8]>)`
///   Adds a `MapArea` to the memory set and optionally copies data into it.
///
/// - `map_trampoline(&mut self)`
///   Maps the trampoline code into the memory set.
///
/// - `new_kernel() -> Self`
///   Creates a new kernel memory set, mapping kernel sections and physical memory.
///
/// - `from_elf(elf_data: &[u8]) -> (Self, usize, usize)`
///   Creates a new memory set from an ELF file, mapping its sections and setting up the user stack and trap context.
///
/// - `activate(&self)`
///   Activates the memory set by writing to the `satp` CSR register and performing a TLB flush.
///
/// - `translate(&self, vpn: VirtPageNum) -> Option<PageTableEntry>`
///   Translates a virtual page number to a page table entry.
///
/// - `shrink_to(&mut self, start: VirtAddr, new_end: VirtAddr) -> bool`
///   Shrinks a memory area to a new end address.
///
/// - `append_to(&mut self, start: VirtAddr, new_end: VirtAddr) -> bool`
///   Extends a memory area to a new end address.
///
/// - `map(&mut self, start: VirtAddr, end: VirtAddr, permission: MapPermission) -> Result<(), ()>`
///   Maps a new area in the memory set, returning an error if there are conflicts.
///
/// - `unmap(&mut self, start: VirtAddr, end: VirtAddr) -> Result<(), ()>`
///   Unmaps an area in the memory set, returning an error if there are conflicts.
///
/// - `map_one(&mut self, vpn: VirtPageNum, ppn: PhysPageNum, map_perm: MapPermission)`
///   Maps a single virtual page to a physical page.
///
/// - `unmap_one(&mut self, vpn: VirtPageNum)`
///   Unmaps a single virtual page.
///
/// The `MapArea` struct represents a contiguous piece of virtual memory and provides methods to manage it.
///
/// # Methods
///
/// - `new(start_va: VirtAddr, end_va: VirtAddr, map_type: MapType, map_perm: MapPermission) -> Self`
///   Creates a new `MapArea`.
///
/// - `map_one(&mut self, page_table: &mut PageTable, vpn: VirtPageNum)`
///   Maps a single virtual page in the `MapArea`.
///
/// - `unmap_one(&mut self, page_table: &mut PageTable, vpn: VirtPageNum)`
///   Unmaps a single virtual page in the `MapArea`.
///
/// - `map(&mut self, page_table: &mut PageTable)`
///   Maps all virtual pages in the `MapArea`.
///
/// - `unmap(&mut self, page_table: &mut PageTable)`
///   Unmaps all virtual pages in the `MapArea`.
///
/// - `shrink_to(&mut self, page_table: &mut PageTable, new_end: VirtPageNum)`
///   Shrinks the `MapArea` to a new end page number.
///
/// - `shrink_from_begin(&mut self, page_table: &mut PageTable, new_start: VirtPageNum)`
///   Shrinks the `MapArea` from the beginning to a new start page number.
///
/// - `shrink_to_slice(&mut self, page_table: &mut PageTable, new_start: VirtPageNum, new_end: VirtPageNum)`
///   Shrinks the `MapArea` to a slice defined by new start and end page numbers.
///
/// - `append_to(&mut self, page_table: &mut PageTable, new_end: VirtPageNum)`
///   Extends the `MapArea` to a new end page number.
///
/// - `copy_data(&mut self, page_table: &mut PageTable, data: &[u8])`
///   Copies data into the `MapArea`.
///
/// - `split_at(&mut self, index: VirtPageNum) -> MapArea`
///   Splits the `MapArea` at a given page number, returning the new `MapArea` containing the split-off part.
///
/// The `MapType` enum represents the type of mapping for a memory set, either identical or framed.
///
/// The `MapPermission` struct represents the permissions for a memory set, corresponding to the `R`, `W`, `X`, and `U` flags in a page table entry.
///
/// # Functions
///
/// - `kernel_stack_position(app_id: usize) -> (usize, usize)`
///   Returns the bottom and top addresses of a kernel stack in kernel space.
///
/// - `remap_test()`
///   Performs a remap test in kernel space.
//! Implementation of [`MapArea`] and [`MemorySet`].

use super::{frame_alloc, FrameTracker};
use super::{PTEFlags, PageTable, PageTableEntry};
use super::{PhysAddr, PhysPageNum, VirtAddr, VirtPageNum};
use super::{StepByOne, VPNRange};
use crate::config::{
    KERNEL_STACK_SIZE, MEMORY_END, PAGE_SIZE, TRAMPOLINE, TRAP_CONTEXT_BASE, USER_STACK_SIZE,
};
use crate::sync::UPSafeCell;
use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use alloc::vec::Vec;
use alloc::vec;
use core::arch::asm;
use lazy_static::*;
use riscv::register::satp;

extern "C" {
    fn stext();
    fn etext();
    fn srodata();
    fn erodata();
    fn sdata();
    fn edata();
    fn sbss_with_stack();
    fn ebss();
    fn ekernel();
    fn strampoline();
}

lazy_static! {
    /// The kernel's initial memory mapping(kernel address space)
    pub static ref KERNEL_SPACE: Arc<UPSafeCell<MemorySet>> =
        Arc::new(unsafe { UPSafeCell::new(MemorySet::new_kernel()) });
}
/// address space
pub struct MemorySet {
    page_table: PageTable,
    areas: Vec<MapArea>,
}

impl MemorySet {
    /// Create a new empty `MemorySet`.
    pub fn new_bare() -> Self {
        Self {
            page_table: PageTable::new(),
            areas: Vec::new(),
        }
    }
    /// Get the page table token
    pub fn token(&self) -> usize {
        self.page_table.token()
    }
    /// Assume that no conflicts.
    pub fn insert_framed_area(
        &mut self,
        start_va: VirtAddr,
        end_va: VirtAddr,
        permission: MapPermission,
    ) {
        self.push(
            MapArea::new(start_va, end_va, MapType::Framed, permission),
            None,
        );
    }
    fn push(&mut self, mut map_area: MapArea, data: Option<&[u8]>) {
        map_area.map(&mut self.page_table);
        if let Some(data) = data {
            map_area.copy_data(&mut self.page_table, data);
        }
        self.areas.push(map_area);
    }
    /// Mention that trampoline is not collected by areas.
    fn map_trampoline(&mut self) {
        self.page_table.map(
            VirtAddr::from(TRAMPOLINE).into(),
            PhysAddr::from(strampoline as usize).into(),
            PTEFlags::R | PTEFlags::X,
        );
    }
    /// Without kernel stacks.
    pub fn new_kernel() -> Self {
        let mut memory_set = Self::new_bare();
        // map trampoline
        memory_set.map_trampoline();
        // map kernel sections
        info!(".text [{:#x}, {:#x})", stext as usize, etext as usize);
        info!(".rodata [{:#x}, {:#x})", srodata as usize, erodata as usize);
        info!(".data [{:#x}, {:#x})", sdata as usize, edata as usize);
        info!(
            ".bss [{:#x}, {:#x})",
            sbss_with_stack as usize, ebss as usize
        );
        info!("mapping .text section");
        memory_set.push(
            MapArea::new(
                (stext as usize).into(),
                (etext as usize).into(),
                MapType::Identical,
                MapPermission::R | MapPermission::X,
            ),
            None,
        );
        info!("mapping .rodata section");
        memory_set.push(
            MapArea::new(
                (srodata as usize).into(),
                (erodata as usize).into(),
                MapType::Identical,
                MapPermission::R,
            ),
            None,
        );
        info!("mapping .data section");
        memory_set.push(
            MapArea::new(
                (sdata as usize).into(),
                (edata as usize).into(),
                MapType::Identical,
                MapPermission::R | MapPermission::W,
            ),
            None,
        );
        info!("mapping .bss section");
        memory_set.push(
            MapArea::new(
                (sbss_with_stack as usize).into(),
                (ebss as usize).into(),
                MapType::Identical,
                MapPermission::R | MapPermission::W,
            ),
            None,
        );
        info!("mapping physical memory");
        memory_set.push(
            MapArea::new(
                (ekernel as usize).into(),
                MEMORY_END.into(),
                MapType::Identical,
                MapPermission::R | MapPermission::W,
            ),
            None,
        );
        memory_set
    }
    /// Include sections in elf and trampoline and TrapContext and user stack,
    /// also returns user_sp_base and entry point.
    pub fn from_elf(elf_data: &[u8]) -> (Self, usize, usize) {
        let mut memory_set = Self::new_bare();
        // map trampoline
        memory_set.map_trampoline();
        // map program headers of elf, with U flag
        let elf = xmas_elf::ElfFile::new(elf_data).unwrap();
        let elf_header = elf.header;
        let magic = elf_header.pt1.magic;
        assert_eq!(magic, [0x7f, 0x45, 0x4c, 0x46], "invalid elf!");
        let ph_count = elf_header.pt2.ph_count();
        let mut max_end_vpn = VirtPageNum(0);
        for i in 0..ph_count {
            let ph = elf.program_header(i).unwrap();
            if ph.get_type().unwrap() == xmas_elf::program::Type::Load {
                let start_va: VirtAddr = (ph.virtual_addr() as usize).into();
                let end_va: VirtAddr = ((ph.virtual_addr() + ph.mem_size()) as usize).into();
                let mut map_perm = MapPermission::U;
                let ph_flags = ph.flags();
                if ph_flags.is_read() {
                    map_perm |= MapPermission::R;
                }
                if ph_flags.is_write() {
                    map_perm |= MapPermission::W;
                }
                if ph_flags.is_execute() {
                    map_perm |= MapPermission::X;
                }
                let map_area = MapArea::new(start_va, end_va, MapType::Framed, map_perm);
                max_end_vpn = map_area.vpn_range.get_end();
                memory_set.push(
                    map_area,
                    Some(&elf.input[ph.offset() as usize..(ph.offset() + ph.file_size()) as usize]),
                );
            }
        }
        // map user stack with U flags
        let max_end_va: VirtAddr = max_end_vpn.into();
        let mut user_stack_bottom: usize = max_end_va.into();
        // guard page
        user_stack_bottom += PAGE_SIZE;
        let user_stack_top = user_stack_bottom + USER_STACK_SIZE;
        memory_set.push(
            MapArea::new(
                user_stack_bottom.into(),
                user_stack_top.into(),
                MapType::Framed,
                MapPermission::R | MapPermission::W | MapPermission::U,
            ),
            None,
        );
        // used in sbrk
        memory_set.push(
            MapArea::new(
                user_stack_top.into(),
                user_stack_top.into(),
                MapType::Framed,
                MapPermission::R | MapPermission::W | MapPermission::U,
            ),
            None,
        );
        // map TrapContext
        memory_set.push(
            MapArea::new(
                TRAP_CONTEXT_BASE.into(),
                TRAMPOLINE.into(),
                MapType::Framed,
                MapPermission::R | MapPermission::W,
            ),
            None,
        );
        (
            memory_set,
            user_stack_top,
            elf.header.pt2.entry_point() as usize,
        )
    }
    /// Change page table by writing satp CSR Register.
    pub fn activate(&self) {
        let satp = self.page_table.token();
        unsafe {
            satp::write(satp);
            asm!("sfence.vma");
        }
    }
    /// Translate a virtual page number to a page table entry
    pub fn translate(&self, vpn: VirtPageNum) -> Option<PageTableEntry> {
        self.page_table.translate(vpn)
    }
    /// shrink the area to new_end
    #[allow(unused)]
    pub fn shrink_to(&mut self, start: VirtAddr, new_end: VirtAddr) -> bool {
        if let Some(area) = self
            .areas
            .iter_mut()
            .find(|area| area.vpn_range.get_start() == start.floor())
        {
            area.shrink_to(&mut self.page_table, new_end.ceil());
            true
        } else {
            false
        }
    }

    /// append the area to new_end
    #[allow(unused)]
    pub fn append_to(&mut self, start: VirtAddr, new_end: VirtAddr) -> bool {
        if let Some(area) = self
            .areas
            .iter_mut()
            .find(|area| area.vpn_range.get_start() == start.floor())
        {
            area.append_to(&mut self.page_table, new_end.ceil());
            true
        } else {
            false
        }
    }

    /// Map a area, return error when conflicts
    pub fn map(&mut self, start: VirtAddr, end: VirtAddr, permission: MapPermission) -> Result<(), ()> {
        for area in &self.areas {
            if !(end <= area.vpn_range.get_start().into()
            || start >= area.vpn_range.get_end().into()) {
                return Err(());
            }
        }

        self.insert_framed_area(start, end, permission);

        Ok(())
    }

    /// Unmap a area, return error when conflicts
    pub fn unmap(&mut self, start: VirtAddr, end: VirtAddr) -> Result<(),()> {
        let mut parts: Vec<(VirtPageNum,VirtPageNum)> = vec![(start.floor(), end.ceil())];
        let mut remove_index: Vec<usize> = Vec::new();
        let mut shrink_from_begin: Vec<(usize, VirtPageNum)> = Vec::new();
        let mut shrink_from_end: Vec<(usize, VirtPageNum)> = Vec::new();
        let mut shrink: Vec<(usize, VirtPageNum, VirtPageNum)> = Vec::new();

        for (i, area) in self.areas.iter().enumerate() {
            // area.data_frames.
            for (j, part) in parts.clone().iter().enumerate() {
                // if contains a range that was 
                if part.0 == area.vpn_range.get_start() && part.1 == area.vpn_range.get_end() {
                    remove_index.push(i);
                    parts.remove(j);
                    break;
                }

                if part.0 > area.vpn_range.get_start() && part.1 >= area.vpn_range.get_end() && part.0 < area.vpn_range.get_end() {
                    shrink_from_end.push((i, part.0));
                    parts.remove(j);
                    if part.1 > area.vpn_range.get_end() {
                        parts.push((area.vpn_range.get_end(),part.1));
                    }
                    break;
                }

                if part.0 <= area.vpn_range.get_start() && part.1 < area.vpn_range.get_end() && part.1 > area.vpn_range.get_start() {
                    shrink_from_begin.push((i, part.1));
                    parts.remove(j);
                    if part.0 < area.vpn_range.get_start() {
                        parts.push((part.0,area.vpn_range.get_start()));
                    }
                    break;
                }

                if part.0 > area.vpn_range.get_start() && part.0 < area.vpn_range.get_end() 
                && part.1 > area.vpn_range.get_start() && part.1 < area.vpn_range.get_end() {
                    shrink.push((i, part.0, part.1));
                    parts.remove(j);
                    break;
                }
            }
        }

        if !parts.is_empty() {
            return Err(())
        }

        // shrink all map areas that required to shrink
        for it in shrink_from_begin {
            self.areas[it.0].shrink_from_begin(&mut self.page_table, it.1);
        }

        for it in shrink_from_end {
            self.areas[it.0].shrink_to(&mut self.page_table, it.1);
        }

        for it in shrink {
            self.areas[it.0].shrink_to_slice(&mut self.page_table, it.1, it.2);
        }

        // remove all map areas that required to remove
        let mut i: usize = 0;
        for index in remove_index.clone() {
            self.areas[index].unmap(&mut self.page_table);
        }
        self.areas.retain(|_| {
            i += 1;
            !remove_index.contains(&(i - 1))
        });

        Ok(())
    }

    /// map one virtual page, may cause leaking in pte.
    pub fn map_one(&mut self, vpn: VirtPageNum, ppn: PhysPageNum, map_perm: MapPermission) {
        let flags = PTEFlags::from_bits(map_perm.bits).unwrap();
        self.page_table.map(vpn, ppn, flags);
    }

    /// map one virtual page, may cause leaking in pte.
    pub fn unmap_one(&mut self, vpn: VirtPageNum) {
        self.page_table.unmap(vpn);
    }
}
/// map area structure, controls a contiguous piece of virtual memory
pub struct MapArea {
    vpn_range: VPNRange,
    data_frames: BTreeMap<VirtPageNum, FrameTracker>,
    map_type: MapType,
    map_perm: MapPermission,
}

impl MapArea {
    pub fn new(
        start_va: VirtAddr,
        end_va: VirtAddr,
        map_type: MapType,
        map_perm: MapPermission,
    ) -> Self {
        let start_vpn: VirtPageNum = start_va.floor();
        let end_vpn: VirtPageNum = end_va.ceil();
        Self {
            vpn_range: VPNRange::new(start_vpn, end_vpn),
            data_frames: BTreeMap::new(),
            map_type,
            map_perm,
        }
    }
    pub fn map_one(&mut self, page_table: &mut PageTable, vpn: VirtPageNum) {
        let ppn: PhysPageNum;
        match self.map_type {
            MapType::Identical => {
                ppn = PhysPageNum(vpn.0);
            }
            MapType::Framed => {
                let frame = frame_alloc().unwrap();
                ppn = frame.ppn;
                self.data_frames.insert(vpn, frame);
            }
        }
        let pte_flags = PTEFlags::from_bits(self.map_perm.bits).unwrap();
        page_table.map(vpn, ppn, pte_flags);
    }
    #[allow(unused)]
    pub fn unmap_one(&mut self, page_table: &mut PageTable, vpn: VirtPageNum) {
        if self.map_type == MapType::Framed {
            self.data_frames.remove(&vpn);
        }
        page_table.unmap(vpn);
    }
    pub fn map(&mut self, page_table: &mut PageTable) {
        for vpn in self.vpn_range {
            self.map_one(page_table, vpn);
        }
    }
    #[allow(unused)]
    pub fn unmap(&mut self, page_table: &mut PageTable) {
        for vpn in self.vpn_range {
            self.unmap_one(page_table, vpn);
        }
    }
    #[allow(unused)]
    pub fn shrink_to(&mut self, page_table: &mut PageTable, new_end: VirtPageNum) {
        for vpn in VPNRange::new(new_end, self.vpn_range.get_end()) {
            self.unmap_one(page_table, vpn)
        }
        self.vpn_range = VPNRange::new(self.vpn_range.get_start(), new_end);
    }
    #[allow(unused)]
    pub fn shrink_from_begin(&mut self, page_table: &mut PageTable, new_start: VirtPageNum) {
        for vpn in VPNRange::new(self.vpn_range.get_start(), new_start) {
            self.unmap_one(page_table, vpn)
        }
        self.vpn_range = VPNRange::new(new_start, self.vpn_range.get_end());
    }
    #[allow(unused)]
    pub fn shrink_to_slice(&mut self, page_table: &mut PageTable, new_start: VirtPageNum, new_end: VirtPageNum) {
        for vpn in VPNRange::new(self.vpn_range.get_start(), new_start) {
            self.unmap_one(page_table, vpn)
        }
        for vpn in VPNRange::new(new_end, self.vpn_range.get_end()) {
            self.unmap_one(page_table, vpn)
        }
        self.vpn_range = VPNRange::new(new_start, new_end);
    }
    #[allow(unused)]
    pub fn append_to(&mut self, page_table: &mut PageTable, new_end: VirtPageNum) {
        for vpn in VPNRange::new(self.vpn_range.get_end(), new_end) {
            self.map_one(page_table, vpn)
        }
        self.vpn_range = VPNRange::new(self.vpn_range.get_start(), new_end);
    }
    /// data: start-aligned but maybe with shorter length
    /// assume that all frames were cleared before
    pub fn copy_data(&mut self, page_table: &mut PageTable, data: &[u8]) {
        assert_eq!(self.map_type, MapType::Framed);
        let mut start: usize = 0;
        let mut current_vpn = self.vpn_range.get_start();
        let len = data.len();
        loop {
            let src = &data[start..len.min(start + PAGE_SIZE)];
            let dst = &mut page_table
                .translate(current_vpn)
                .unwrap()
                .ppn()
                .get_bytes_array()[..src.len()];
            dst.copy_from_slice(src);
            start += PAGE_SIZE;
            if start >= len {
                break;
            }
            current_vpn.step();
        }
    }

    /// Split the map area into two pieces.
    /// current map area will be the shrinked,
    /// returned map area will contain shrinked part.
    #[allow(unused)]
    pub fn split_at(&mut self, index: VirtPageNum) -> MapArea{
        assert!(index < self.vpn_range.get_end(), "Given vpn range was longer than end.");
        assert!(index > self.vpn_range.get_start(), "Given vpn range was shorter than start.");

        let ret = MapArea{
            vpn_range: VPNRange::new(index, self.vpn_range.get_end()),
            data_frames: self.data_frames.split_off(&index),
            map_type: self.map_type,
            map_perm: self.map_perm,
        };
        
        self.vpn_range = VPNRange::new(self.vpn_range.get_start(), index);
        ret
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
/// map type for memory set: identical or framed
pub enum MapType {
    Identical,
    Framed,
}

bitflags! {
    /// map permission corresponding to that in pte: `R W X U`
    pub struct MapPermission: u8 {
        ///Readable
        const R = 1 << 1;
        ///Writable
        const W = 1 << 2;
        ///Excutable
        const X = 1 << 3;
        ///Accessible in U mode
        const U = 1 << 4;
    }
}

/// Return (bottom, top) of a kernel stack in kernel space.
pub fn kernel_stack_position(app_id: usize) -> (usize, usize) {
    let top = TRAMPOLINE - app_id * (KERNEL_STACK_SIZE + PAGE_SIZE);
    let bottom = top - KERNEL_STACK_SIZE;
    (bottom, top)
}

/// remap test in kernel space
#[allow(unused)]
pub fn remap_test() {
    let mut kernel_space = KERNEL_SPACE.exclusive_access();
    let mid_text: VirtAddr = ((stext as usize + etext as usize) / 2).into();
    let mid_rodata: VirtAddr = ((srodata as usize + erodata as usize) / 2).into();
    let mid_data: VirtAddr = ((sdata as usize + edata as usize) / 2).into();
    assert!(!kernel_space
        .page_table
        .translate(mid_text.floor())
        .unwrap()
        .writable(),);
    assert!(!kernel_space
        .page_table
        .translate(mid_rodata.floor())
        .unwrap()
        .writable(),);
    assert!(!kernel_space
        .page_table
        .translate(mid_data.floor())
        .unwrap()
        .executable(),);
    println!("remap_test passed!");
}
