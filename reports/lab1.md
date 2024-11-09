# 编程作业

## 获取任务信息

基本思路是在 TASK_MANAGER 处开个洞，加一个给对应任务统计系统调用次数的接口。

然后在 syscall 的入口处加入调用 increase syscall 的操作。（直接在所有 syscall 里累加也行但是太蠢了。）

syscall 的统计到此结束。

然后是每个任务的运行时间，TaskInfo 里的 time 字段。这里我在每次任务第一次调度运行时记录下时间，然后在这个 syscall 里返回当前时间减去任务时间。

切记 **不要使用 os 里的 get_time**，要用 crate::timer 中实现的。否则无法通过。

另外，任务可能会被不断 switch & run，所以不要以为在里边初始化 begin_time 就 ok 了。初始化前务必检查它是否已有有效值！

~~往ci-user的bin里加了println!输出调试信息没删 会影响本地测试时WRITE系统调用次数和预期的不一样 导致测试通不过 大伙要是加了记得make test之前删了...~~

# 简答作业

1. 正确进入 U 态后，程序的特征还应有：使用 S 态特权指令，访问 S 态寄存器后会报错。 请同学们可以自行测试这些内容（运行 三个 bad 测例 (ch2b_bad_*.rs) ）， 描述程序出错行为，同时注意注明你使用的 sbi 及其版本。

观察到默认进行 ci-user make 时，这些 ch2b_bad_*.rs 已经被编译嵌入系统中。应当会被执行。因此直接通过 make test CHAPTER=3 执行，观察 LOG。

```bash
[build.py] application ch2b_bad_address start with address 0x80400000
   Compiling user_lib v0.1.0 (/home/ezra/2024s-rcore-255doesnotexist/ci-user/user)
    Finished `release` profile [optimized] target(s) in 0.33s
[build.py] application ch2b_bad_instructions start with address 0x80420000
   Compiling user_lib v0.1.0 (/home/ezra/2024s-rcore-255doesnotexist/ci-user/user)
    Finished `release` profile [optimized] target(s) in 0.33s
[build.py] application ch2b_bad_register start with address 0x80440000
   Compiling user_lib v0.1.0 (/home/ezra/2024s-rcore-255doesnotexist/ci-user/user)
    Finished `release` profile [optimized] target(s) in 0.33s
[build.py] application ch2b_hello_world start with address 0x80460000
   Compiling user_lib v0.1.0 (/home/ezra/2024s-rcore-255doesnotexist/ci-user/user)
    Finished `release` profile [optimized] target(s) in 0.36s
```

```bash
[kernel] Hello, world!
[kernel] PageFault in application, bad addr = 0x0, bad instruction = 0x804003ac, kernel killed it.
[kernel] IllegalInstruction in application, kernel killed it.
[kernel] IllegalInstruction in application, kernel killed it.
Hello, world from user mode program!
```

可见在成功执行 Hello World (user mode) 前出现的页面错误、非法指令应该就是这些 bad 指令产生的结果。

我产生这些结果使用的 RustSBI 版本信息如下。

```bash
timeout --foreground 30s qemu-system-riscv64 \
        -machine virt \
        -nographic \
        -bios ../bootloader/rustsbi-qemu.bin \
        -kernel target/riscv64gc-unknown-none-elf/release/os
[rustsbi] RustSBI version 0.3.0-alpha.2, adapting to RISC-V SBI v1.0.0
.______       __    __      _______.___________.  _______..______   __
|   _  \     |  |  |  |    /       |           | /       ||   _  \ |  |
|  |_)  |    |  |  |  |   |   (----`---|  |----`|   (----`|  |_)  ||  |
|      /     |  |  |  |    \   \       |  |      \   \    |   _  < |  |
|  |\  \----.|  `--'  |.----)   |      |  |  .----)   |   |  |_)  ||  |
| _| `._____| \______/ |_______/       |__|  |_______/    |______/ |__|
[rustsbi] Implementation     : RustSBI-QEMU Version 0.2.0-alpha.2
[rustsbi] Platform Name      : riscv-virtio,qemu
[rustsbi] Platform SMP       : 1
[rustsbi] Platform Memory    : 0x80000000..0x88000000
[rustsbi] Boot HART          : 0
[rustsbi] Device Tree Region : 0x87000000..0x87000ef2
[rustsbi] Firmware Address   : 0x80000000
[rustsbi] Supervisor Address : 0x80200000
[rustsbi] pmp01: 0x00000000..0x80000000 (-wr)
[rustsbi] pmp02: 0x80000000..0x80200000 (---)
[rustsbi] pmp03: 0x80200000..0x88000000 (xwr)
[rustsbi] pmp04: 0x88000000..0x00000000 (-wr)
```

2. 深入理解 trap.S 中两个函数 __alltraps 和 __restore 的作用，并回答如下问题:

    1. L40：刚进入 __restore 时，a0 代表了什么值。请指出 __restore 的两种使用情景。

    刚刚进入 __restore 时，a0 代表了待恢复内核栈的内存地址。运行 __restore 是为了转移 sp 到一个新的内核栈上。

    它既可以用来启动一个新的应用程序，也可以用来在 Trap 处理完成后恢复现场。

    2. L43-L48：这几行汇编代码特殊处理了哪些寄存器？这些寄存器的的值对于进入用户态有何意义？请分别解释。

    ```asm
    ld t0, 32*8(sp) # 这三行是加载原本的信息，便于恢复
    ld t1, 33*8(sp)
    ld t2, 2*8(sp)
    csrw sstatus, t0 # 恢复用户态信息，保证在用户态下执行
    csrw sepc, t1 # 确保用户程序执行中断后能返回到哪条指令
    csrw sscratch, t2 # 提供恢复关键寄存器值的上下文
    ```

    这里恢复的是 CSR 寄存器。这些寄存器是用来控制 CPU 各类控制和状态的。
    具体的分别解释我写题面上代码注释里咯。总之就是从 Trap 栈里读取恢复信息。

    3. L50-L56：为何跳过了 x2 和 x4？

    ```asm
    ld x1, 1*8(sp)
    ld x3, 3*8(sp)
    .set n, 5
    .rept 27
    LOAD_GP %n
    .set n, n+1
    .endr
    ```

    因为 x2 是 sp、x4 是 tp。他们不属于通用寄存器，控制了控制流。
    如果需要更改，也不该在这里更改。

    4. L60：该指令之后，sp 和 sscratch 中的值分别有什么意义？

    ```asm
    csrrw sp, sscratch, sp
    ```

    这条指令后，sp 指向了内核栈、sscratch 指向了用户栈。这是一次交换。

    5. __restore：中发生状态切换在哪一条指令？为何该指令执行之后会进入用户态？

    sret。它会从 sepc 寄存器中恢复 PC，并将 CPU 的模式从 S 模式切换回 U 模式。

    6. L13：该指令之后，sp 和 sscratch 中的值分别有什么意义？

    ```asm
    csrrw sp, sscratch, sp
    ```

    刚刚还觉得迷惑这不是问过了吗，才发现原来前面还有行号，我倒。

    这里的 sp 指向内核栈，sscratch 指向用户栈。（注释里不是写了吗啊啊！！）

    7. 从 U 态进入 S 态是哪一条指令发生的？

    ```csrrw sp, sscratch, sp``` 指令交换了 sp 和 sscratch（备用栈指针）寄存器的值。这个操作是 U 态到 S 态转换的关键步骤，它使得 CPU 开始使用内核栈，并将用户栈的地址保存在 sscratch 中。

# 荣誉准则

在完成本次实验的过程（含此前学习的过程）中，我曾分别与以下各位就（与本次实验相关的）以下方面做过交流，还在代码中对应的位置以注释形式记录了具体的交流对象及内容：

- KamijiToma

交流内容：

2> 关于 task time 的 test 一直过不去 我感觉逻辑是对的啊

K> 不知道 出来跟我吃饭 二楼西那个餐厅 吃带汤汁的烤盘饭 我在12组门口了

2> ok 我穿衣服出来 log 看起来拿到的时间没问题呀

K> 我在南边的柳树下面 下雨了

(吃完回宿舍后)

2> 哎哟 我调用的计时器不对 get time us 是 os 提供的 应该用 crate 提供的 timer

此外，我也参考了 以下资料 ，还在代码中对应的位置以注释形式记录了具体的参考来源及内容：

训练营资料：
- https://learningos.cn/rCore-Tutorial-Guide-2024S/

原版 rCore 文档资料及其评论区：
- https://rcore-os.cn/rCore-Tutorial-Book-v3/chapter3/index.html

3. 我独立完成了本次实验除以上方面之外的所有工作，包括代码与文档。 我清楚地知道，从以上方面获得的信息在一定程度上降低了实验难度，可能会影响起评分。

4. 我从未使用过他人的代码，不管是原封不动地复制，还是经过了某些等价转换。 我未曾也不会向他人（含此后各届同学）复制或公开我的实验代码，我有义务妥善保管好它们。 我提交至本实验的评测系统的代码，均无意于破坏或妨碍任何计算机系统的正常运转。 我清楚地知道，以上情况均为本课程纪律所禁止，若违反，对应的实验成绩将按“-100”分计。