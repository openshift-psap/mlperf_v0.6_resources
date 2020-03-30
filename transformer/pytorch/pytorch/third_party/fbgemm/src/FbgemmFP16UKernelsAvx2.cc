/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "FbgemmFP16UKernelsAvx2.h"

namespace fbgemm {

void __attribute__((noinline)) gemmkernel_1x2_AVX2_fA0fB0fC0(GemmParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // k
      "mov r8, [r14 + 0]\t\n"
      // A
      "mov r9, [r14 + 8]\t\n"
      // B
      "mov r10, [r14 + 16]\t\n"
      // beta
      "mov r15, [r14 + 24]\t\n"
      // accum
      "mov rdx, [r14 + 32]\t\n"
      // C
      "mov r12, [r14 + 40]\t\n"
      // ldc
      "mov r13, [r14 + 48]\t\n"
      // b_block_cols
      "mov rdi, [r14 + 56]\t\n"
      // b_block_size
      "mov rsi, [r14 + 64]\t\n"
      // Make copies of A and C
      "mov rax, r9\t\n"
      "mov rcx, r12\t\n"

      "mov rbx, 0\t\n"
      "loop_outter%=:\t\n"
      "mov r14, 0\t\n"
      "vxorps ymm0,ymm0,ymm0\t\n"
      "vxorps ymm1,ymm1,ymm1\t\n"


      "loop_inner%=:\t\n"

      "vcvtph2ps ymm3,XMMWORD PTR [r10 + 0]\t\n"
      "vcvtph2ps ymm4,XMMWORD PTR [r10 + 16]\t\n"
      "vbroadcastss ymm2,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm3,ymm2\t\n"
      "vfmadd231ps ymm1,ymm4,ymm2\t\n"
      "add r9,4\t\n"
      "add r10,32\t\n"
      "inc r14\t\n"
      "cmp r14, r8\t\n"
      "jl loop_inner%=\t\n"

      "L_exit%=:\t\n"

      "cmp rdx, 1\t\n"
      "je L_accum%=\t\n"
      // Dump C
      "vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "jmp L_done%=\t\n"

      "L_accum%=:\t\n"
      // Dump C with accumulate
      "vbroadcastss ymm15,DWORD PTR [r15]\t\n"
      "vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
      "vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"

      "L_done%=:\t\n"

      // next outer iteration
      "add rcx, 64\t\n"
      "mov r12, rcx\t\n"
      "mov r9, rax\t\n"
      "inc rbx\t\n"
      "cmp rbx, rdi\t\n"
      "jl loop_outter%=\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r15",
        "r13",
        "r14",
        "rax",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "rbx",
        "r12",
        "memory");
}
void __attribute__((noinline)) gemmkernel_2x2_AVX2_fA0fB0fC0(GemmParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // k
      "mov r8, [r14 + 0]\t\n"
      // A
      "mov r9, [r14 + 8]\t\n"
      // B
      "mov r10, [r14 + 16]\t\n"
      // beta
      "mov r15, [r14 + 24]\t\n"
      // accum
      "mov rdx, [r14 + 32]\t\n"
      // C
      "mov r12, [r14 + 40]\t\n"
      // ldc
      "mov r13, [r14 + 48]\t\n"
      // b_block_cols
      "mov rdi, [r14 + 56]\t\n"
      // b_block_size
      "mov rsi, [r14 + 64]\t\n"
      // Make copies of A and C
      "mov rax, r9\t\n"
      "mov rcx, r12\t\n"

      "mov rbx, 0\t\n"
      "loop_outter%=:\t\n"
      "mov r14, 0\t\n"
      "vxorps ymm0,ymm0,ymm0\t\n"
      "vxorps ymm1,ymm1,ymm1\t\n"
      "vxorps ymm2,ymm2,ymm2\t\n"
      "vxorps ymm3,ymm3,ymm3\t\n"


      "loop_inner%=:\t\n"

      "vcvtph2ps ymm5,XMMWORD PTR [r10 + 0]\t\n"
      "vcvtph2ps ymm6,XMMWORD PTR [r10 + 16]\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm5,ymm4\t\n"
      "vfmadd231ps ymm1,ymm6,ymm4\t\n"
      "vbroadcastss ymm4,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm5,ymm4\t\n"
      "vfmadd231ps ymm3,ymm6,ymm4\t\n"
      "add r9,8\t\n"
      "add r10,32\t\n"
      "inc r14\t\n"
      "cmp r14, r8\t\n"
      "jl loop_inner%=\t\n"

      "L_exit%=:\t\n"

      "cmp rdx, 1\t\n"
      "je L_accum%=\t\n"
      // Dump C
      "vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
      "add r12, r13\t\n"
      "jmp L_done%=\t\n"

      "L_accum%=:\t\n"
      // Dump C with accumulate
      "vbroadcastss ymm15,DWORD PTR [r15]\t\n"
      "vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
      "vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
      "vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
      "add r12, r13\t\n"

      "L_done%=:\t\n"

      // next outer iteration
      "add rcx, 64\t\n"
      "mov r12, rcx\t\n"
      "mov r9, rax\t\n"
      "inc rbx\t\n"
      "cmp rbx, rdi\t\n"
      "jl loop_outter%=\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r15",
        "r13",
        "r14",
        "rax",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "rbx",
        "r12",
        "memory");
}
void __attribute__((noinline)) gemmkernel_3x2_AVX2_fA0fB0fC0(GemmParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // k
      "mov r8, [r14 + 0]\t\n"
      // A
      "mov r9, [r14 + 8]\t\n"
      // B
      "mov r10, [r14 + 16]\t\n"
      // beta
      "mov r15, [r14 + 24]\t\n"
      // accum
      "mov rdx, [r14 + 32]\t\n"
      // C
      "mov r12, [r14 + 40]\t\n"
      // ldc
      "mov r13, [r14 + 48]\t\n"
      // b_block_cols
      "mov rdi, [r14 + 56]\t\n"
      // b_block_size
      "mov rsi, [r14 + 64]\t\n"
      // Make copies of A and C
      "mov rax, r9\t\n"
      "mov rcx, r12\t\n"

      "mov rbx, 0\t\n"
      "loop_outter%=:\t\n"
      "mov r14, 0\t\n"
      "vxorps ymm0,ymm0,ymm0\t\n"
      "vxorps ymm1,ymm1,ymm1\t\n"
      "vxorps ymm2,ymm2,ymm2\t\n"
      "vxorps ymm3,ymm3,ymm3\t\n"
      "vxorps ymm4,ymm4,ymm4\t\n"
      "vxorps ymm5,ymm5,ymm5\t\n"


      "loop_inner%=:\t\n"

      "vcvtph2ps ymm7,XMMWORD PTR [r10 + 0]\t\n"
      "vcvtph2ps ymm8,XMMWORD PTR [r10 + 16]\t\n"
      "vbroadcastss ymm6,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm7,ymm6\t\n"
      "vfmadd231ps ymm1,ymm8,ymm6\t\n"
      "vbroadcastss ymm6,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm7,ymm6\t\n"
      "vfmadd231ps ymm3,ymm8,ymm6\t\n"
      "vbroadcastss ymm6,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm7,ymm6\t\n"
      "vfmadd231ps ymm5,ymm8,ymm6\t\n"
      "add r9,12\t\n"
      "add r10,32\t\n"
      "inc r14\t\n"
      "cmp r14, r8\t\n"
      "jl loop_inner%=\t\n"

      "L_exit%=:\t\n"

      "cmp rdx, 1\t\n"
      "je L_accum%=\t\n"
      // Dump C
      "vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
      "add r12, r13\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
      "add r12, r13\t\n"
      "jmp L_done%=\t\n"

      "L_accum%=:\t\n"
      // Dump C with accumulate
      "vbroadcastss ymm15,DWORD PTR [r15]\t\n"
      "vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
      "vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
      "vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
      "add r12, r13\t\n"
      "vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
      "vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
      "add r12, r13\t\n"

      "L_done%=:\t\n"

      // next outer iteration
      "add rcx, 64\t\n"
      "mov r12, rcx\t\n"
      "mov r9, rax\t\n"
      "inc rbx\t\n"
      "cmp rbx, rdi\t\n"
      "jl loop_outter%=\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r15",
        "r13",
        "r14",
        "rax",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "rbx",
        "r12",
        "memory");
}
void __attribute__((noinline)) gemmkernel_4x2_AVX2_fA0fB0fC0(GemmParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // k
      "mov r8, [r14 + 0]\t\n"
      // A
      "mov r9, [r14 + 8]\t\n"
      // B
      "mov r10, [r14 + 16]\t\n"
      // beta
      "mov r15, [r14 + 24]\t\n"
      // accum
      "mov rdx, [r14 + 32]\t\n"
      // C
      "mov r12, [r14 + 40]\t\n"
      // ldc
      "mov r13, [r14 + 48]\t\n"
      // b_block_cols
      "mov rdi, [r14 + 56]\t\n"
      // b_block_size
      "mov rsi, [r14 + 64]\t\n"
      // Make copies of A and C
      "mov rax, r9\t\n"
      "mov rcx, r12\t\n"

      "mov rbx, 0\t\n"
      "loop_outter%=:\t\n"
      "mov r14, 0\t\n"
      "vxorps ymm0,ymm0,ymm0\t\n"
      "vxorps ymm1,ymm1,ymm1\t\n"
      "vxorps ymm2,ymm2,ymm2\t\n"
      "vxorps ymm3,ymm3,ymm3\t\n"
      "vxorps ymm4,ymm4,ymm4\t\n"
      "vxorps ymm5,ymm5,ymm5\t\n"
      "vxorps ymm6,ymm6,ymm6\t\n"
      "vxorps ymm7,ymm7,ymm7\t\n"


      "loop_inner%=:\t\n"

      "vcvtph2ps ymm9,XMMWORD PTR [r10 + 0]\t\n"
      "vcvtph2ps ymm10,XMMWORD PTR [r10 + 16]\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm9,ymm8\t\n"
      "vfmadd231ps ymm1,ymm10,ymm8\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm9,ymm8\t\n"
      "vfmadd231ps ymm3,ymm10,ymm8\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm9,ymm8\t\n"
      "vfmadd231ps ymm5,ymm10,ymm8\t\n"
      "vbroadcastss ymm8,DWORD PTR [r9+12]\t\n"
      "vfmadd231ps ymm6,ymm9,ymm8\t\n"
      "vfmadd231ps ymm7,ymm10,ymm8\t\n"
      "add r9,16\t\n"
      "add r10,32\t\n"
      "inc r14\t\n"
      "cmp r14, r8\t\n"
      "jl loop_inner%=\t\n"

      "L_exit%=:\t\n"

      "cmp rdx, 1\t\n"
      "je L_accum%=\t\n"
      // Dump C
      "vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
      "add r12, r13\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
      "add r12, r13\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm7\t\n"
      "add r12, r13\t\n"
      "jmp L_done%=\t\n"

      "L_accum%=:\t\n"
      // Dump C with accumulate
      "vbroadcastss ymm15,DWORD PTR [r15]\t\n"
      "vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
      "vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
      "vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
      "add r12, r13\t\n"
      "vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
      "vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
      "add r12, r13\t\n"
      "vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
      "vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm7\t\n"
      "add r12, r13\t\n"

      "L_done%=:\t\n"

      // next outer iteration
      "add rcx, 64\t\n"
      "mov r12, rcx\t\n"
      "mov r9, rax\t\n"
      "inc rbx\t\n"
      "cmp rbx, rdi\t\n"
      "jl loop_outter%=\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r15",
        "r13",
        "r14",
        "rax",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "rbx",
        "r12",
        "memory");
}
void __attribute__((noinline)) gemmkernel_5x2_AVX2_fA0fB0fC0(GemmParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // k
      "mov r8, [r14 + 0]\t\n"
      // A
      "mov r9, [r14 + 8]\t\n"
      // B
      "mov r10, [r14 + 16]\t\n"
      // beta
      "mov r15, [r14 + 24]\t\n"
      // accum
      "mov rdx, [r14 + 32]\t\n"
      // C
      "mov r12, [r14 + 40]\t\n"
      // ldc
      "mov r13, [r14 + 48]\t\n"
      // b_block_cols
      "mov rdi, [r14 + 56]\t\n"
      // b_block_size
      "mov rsi, [r14 + 64]\t\n"
      // Make copies of A and C
      "mov rax, r9\t\n"
      "mov rcx, r12\t\n"

      "mov rbx, 0\t\n"
      "loop_outter%=:\t\n"
      "mov r14, 0\t\n"
      "vxorps ymm0,ymm0,ymm0\t\n"
      "vxorps ymm1,ymm1,ymm1\t\n"
      "vxorps ymm2,ymm2,ymm2\t\n"
      "vxorps ymm3,ymm3,ymm3\t\n"
      "vxorps ymm4,ymm4,ymm4\t\n"
      "vxorps ymm5,ymm5,ymm5\t\n"
      "vxorps ymm6,ymm6,ymm6\t\n"
      "vxorps ymm7,ymm7,ymm7\t\n"
      "vxorps ymm8,ymm8,ymm8\t\n"
      "vxorps ymm9,ymm9,ymm9\t\n"


      "loop_inner%=:\t\n"

      "vcvtph2ps ymm11,XMMWORD PTR [r10 + 0]\t\n"
      "vcvtph2ps ymm12,XMMWORD PTR [r10 + 16]\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm11,ymm10\t\n"
      "vfmadd231ps ymm1,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm11,ymm10\t\n"
      "vfmadd231ps ymm3,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm11,ymm10\t\n"
      "vfmadd231ps ymm5,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+12]\t\n"
      "vfmadd231ps ymm6,ymm11,ymm10\t\n"
      "vfmadd231ps ymm7,ymm12,ymm10\t\n"
      "vbroadcastss ymm10,DWORD PTR [r9+16]\t\n"
      "vfmadd231ps ymm8,ymm11,ymm10\t\n"
      "vfmadd231ps ymm9,ymm12,ymm10\t\n"
      "add r9,20\t\n"
      "add r10,32\t\n"
      "inc r14\t\n"
      "cmp r14, r8\t\n"
      "jl loop_inner%=\t\n"

      "L_exit%=:\t\n"

      "cmp rdx, 1\t\n"
      "je L_accum%=\t\n"
      // Dump C
      "vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
      "add r12, r13\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
      "add r12, r13\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm7\t\n"
      "add r12, r13\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm9\t\n"
      "add r12, r13\t\n"
      "jmp L_done%=\t\n"

      "L_accum%=:\t\n"
      // Dump C with accumulate
      "vbroadcastss ymm15,DWORD PTR [r15]\t\n"
      "vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
      "vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
      "vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
      "add r12, r13\t\n"
      "vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
      "vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
      "add r12, r13\t\n"
      "vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
      "vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm7\t\n"
      "add r12, r13\t\n"
      "vfmadd231ps ymm8,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
      "vfmadd231ps ymm9,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm9\t\n"
      "add r12, r13\t\n"

      "L_done%=:\t\n"

      // next outer iteration
      "add rcx, 64\t\n"
      "mov r12, rcx\t\n"
      "mov r9, rax\t\n"
      "inc rbx\t\n"
      "cmp rbx, rdi\t\n"
      "jl loop_outter%=\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r15",
        "r13",
        "r14",
        "rax",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "rbx",
        "r12",
        "memory");
}
void __attribute__((noinline)) gemmkernel_6x2_AVX2_fA0fB0fC0(GemmParams* gp) {
  asm volatile(
#if !defined(__clang__)
      "mov r14, %[gp]\t\n"
#else
      "mov %[gp], %%r14\t\n"
      ".intel_syntax noprefix\t\n"
#endif

      // Copy parameters
      // k
      "mov r8, [r14 + 0]\t\n"
      // A
      "mov r9, [r14 + 8]\t\n"
      // B
      "mov r10, [r14 + 16]\t\n"
      // beta
      "mov r15, [r14 + 24]\t\n"
      // accum
      "mov rdx, [r14 + 32]\t\n"
      // C
      "mov r12, [r14 + 40]\t\n"
      // ldc
      "mov r13, [r14 + 48]\t\n"
      // b_block_cols
      "mov rdi, [r14 + 56]\t\n"
      // b_block_size
      "mov rsi, [r14 + 64]\t\n"
      // Make copies of A and C
      "mov rax, r9\t\n"
      "mov rcx, r12\t\n"

      "mov rbx, 0\t\n"
      "loop_outter%=:\t\n"
      "mov r14, 0\t\n"
      "vxorps ymm0,ymm0,ymm0\t\n"
      "vxorps ymm1,ymm1,ymm1\t\n"
      "vxorps ymm2,ymm2,ymm2\t\n"
      "vxorps ymm3,ymm3,ymm3\t\n"
      "vxorps ymm4,ymm4,ymm4\t\n"
      "vxorps ymm5,ymm5,ymm5\t\n"
      "vxorps ymm6,ymm6,ymm6\t\n"
      "vxorps ymm7,ymm7,ymm7\t\n"
      "vxorps ymm8,ymm8,ymm8\t\n"
      "vxorps ymm9,ymm9,ymm9\t\n"
      "vxorps ymm10,ymm10,ymm10\t\n"
      "vxorps ymm11,ymm11,ymm11\t\n"


      "loop_inner%=:\t\n"

      "vcvtph2ps ymm13,XMMWORD PTR [r10 + 0]\t\n"
      "vcvtph2ps ymm14,XMMWORD PTR [r10 + 16]\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+0]\t\n"
      "vfmadd231ps ymm0,ymm13,ymm12\t\n"
      "vfmadd231ps ymm1,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+4]\t\n"
      "vfmadd231ps ymm2,ymm13,ymm12\t\n"
      "vfmadd231ps ymm3,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+8]\t\n"
      "vfmadd231ps ymm4,ymm13,ymm12\t\n"
      "vfmadd231ps ymm5,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+12]\t\n"
      "vfmadd231ps ymm6,ymm13,ymm12\t\n"
      "vfmadd231ps ymm7,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+16]\t\n"
      "vfmadd231ps ymm8,ymm13,ymm12\t\n"
      "vfmadd231ps ymm9,ymm14,ymm12\t\n"
      "vbroadcastss ymm12,DWORD PTR [r9+20]\t\n"
      "vfmadd231ps ymm10,ymm13,ymm12\t\n"
      "vfmadd231ps ymm11,ymm14,ymm12\t\n"
      "add r9,24\t\n"
      "add r10,32\t\n"
      "inc r14\t\n"
      "cmp r14, r8\t\n"
      "jl loop_inner%=\t\n"

      "L_exit%=:\t\n"

      "cmp rdx, 1\t\n"
      "je L_accum%=\t\n"
      // Dump C
      "vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
      "add r12, r13\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
      "add r12, r13\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm7\t\n"
      "add r12, r13\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm9\t\n"
      "add r12, r13\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm10\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm11\t\n"
      "add r12, r13\t\n"
      "jmp L_done%=\t\n"

      "L_accum%=:\t\n"
      // Dump C with accumulate
      "vbroadcastss ymm15,DWORD PTR [r15]\t\n"
      "vfmadd231ps ymm0,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm0\t\n"
      "vfmadd231ps ymm1,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm1\t\n"
      "add r12, r13\t\n"
      "vfmadd231ps ymm2,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm2\t\n"
      "vfmadd231ps ymm3,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm3\t\n"
      "add r12, r13\t\n"
      "vfmadd231ps ymm4,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm4\t\n"
      "vfmadd231ps ymm5,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm5\t\n"
      "add r12, r13\t\n"
      "vfmadd231ps ymm6,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm6\t\n"
      "vfmadd231ps ymm7,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm7\t\n"
      "add r12, r13\t\n"
      "vfmadd231ps ymm8,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm8\t\n"
      "vfmadd231ps ymm9,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm9\t\n"
      "add r12, r13\t\n"
      "vfmadd231ps ymm10,ymm15,YMMWORD PTR [r12 + 0]\t\n"
      "vmovups YMMWORD PTR [r12 + 0], ymm10\t\n"
      "vfmadd231ps ymm11,ymm15,YMMWORD PTR [r12 + 32]\t\n"
      "vmovups YMMWORD PTR [r12 + 32], ymm11\t\n"
      "add r12, r13\t\n"

      "L_done%=:\t\n"

      // next outer iteration
      "add rcx, 64\t\n"
      "mov r12, rcx\t\n"
      "mov r9, rax\t\n"
      "inc rbx\t\n"
      "cmp rbx, rdi\t\n"
      "jl loop_outter%=\t\n"
      :
      : [gp] "rm"(gp)
      : "r8",
        "r9",
        "r10",
        "r11",
        "r15",
        "r13",
        "r14",
        "rax",
        "rcx",
        "rdx",
        "rsi",
        "rdi",
        "rbx",
        "r12",
        "memory");
}

} // namespace fbgemm
