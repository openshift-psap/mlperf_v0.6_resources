/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <assert.h>
#include <cpuid.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>

using namespace std;

void addi(ofstream& of, string i, bool disable = false) {
  if (disable == false)
    of << "      \"" + i + "\\t\\n\"" + "\n";
}

struct ISA {
  unsigned avx; // 1, 2 or 3
  string name;
  vector<vector<unsigned>> shapes;
};

int main() {
  bool iaca = false;
  bool disable = false;

  bool fixedA = true, fixedB = true, fixedC = true;

  int eax, ebx, ecx, edx;
  __cpuid(1 /* ecx = vendor string */, eax, ebx, ecx, edx);
  printf("FC16 is %s supported\n", ((ecx & bit_F16C) ? " " : "not"));

  string comma = ",";

  vector<ISA> isa = {
      // {1, "AVX", {{4, 1, 0}, {4, 2, 0}, {4, 3, 0}, {3, 1, 0}, {3, 2, 0}, {3,
      // 3, 0}}},
      {2,
       "AVX2",
       {
           // 4x3 register layout
           // {1, 3, 0},
           // {2, 3, 0},
           // {3, 3, 0},
           // {4, 3, 0},

           // 6x2 register layout
           {1, 2, 0},
           {2, 2, 0},
           {3, 2, 0},
           {4, 2, 0},
           {5, 2, 0},
           {6, 2, 0},

           // 14x1 register layout
           // {1, 1, 0},
           // {2, 1, 0},
           // {3, 1, 0},
           // {4, 1, 0},
           // {5, 1, 0},
           // {6, 1, 0},
           // {7, 1, 0},
           // {8, 1, 0},
           // {9, 1, 0},
           // {10, 1, 0},
           // {11, 1, 0},
           // {12, 1, 0},
           // {13, 1, 0},
           // {14, 1, 0},
       }}};

  // open all files
  ofstream srcfile;
  srcfile.open("FbgemmFP16UKernelsAvx2.cc");
  srcfile
      << "/*\n"
         " * Copyright (c) Facebook, Inc. and its affiliates.\n"
         " * All rights reserved.\n"
         " * This source code is licensed under the BSD-style license found in the\n"
         " * LICENSE file in the root directory of this source tree.\n"
         " */\n";
  srcfile << "#include \"FbgemmFP16UKernelsAvx2.h\"\n\n";
  srcfile << "namespace fbgemm {\n\n";
  if (iaca) {
    srcfile << "#include \"iacaMarks.h\"\n";
  }

  ofstream hdrfile;
  hdrfile.open("FbgemmFP16UKernelsAvx2.h");
  hdrfile
      << "/*\n"
         " * Copyright (c) Facebook, Inc. and its affiliates.\n"
         " * All rights reserved.\n"
         " * This source code is licensed under the BSD-style license found in the\n"
         " * LICENSE file in the root directory of this source tree.\n"
         " */\n";

  hdrfile << "#ifndef FBGEMM_UKERNELS\n";
  hdrfile << "#define FBGEMM_UKERNELS\n";
  hdrfile << "#include <cstdint>\n";
  hdrfile << "#include \"fbgemm/Types.h\"\n\n";
  hdrfile << "namespace fbgemm {\n\n";
  hdrfile << "using fp16 = float16;\n";
  hdrfile << "using fp32 = float;\n";
  hdrfile
      << "struct GemmParams {\n  uint64_t k;\n  float* A;\n  const fp16* B;\n"
         "  float* beta;\n  uint64_t accum;\n  float* C;\n  uint64_t ldc;\n"
         "  uint64_t b_block_cols;\n  uint64_t b_block_size;\n};\n";

  std::map<string, string> fptr_typedef;
  fptr_typedef["fp16"] = "";
  fptr_typedef["fp32"] = "";

  unsigned labelId = 0;
#if 1
  for (auto fixedA : {false})
    for (auto fixedB : {false})
      for (auto fixedC : {false})
#else
  for (auto fixedA : {true})
    for (auto fixedB : {true})
      for (auto fixedC : {true})
#endif
        for (auto s : isa) {
          vector<vector<unsigned>>& ukernel_shape = s.shapes;

          vector<string> funcname(ukernel_shape.size()),
              fheader(ukernel_shape.size());
          string fargs;

          for (auto fp16 : {true}) {
            string B_type = ((fp16) ? "fp16" : "fp32");
            string prefix = s.name + /*"_" + B_type */ +"_" + "fA" +
                to_string(fixedA) + "fB" + to_string(fixedB) + "fC" +
                to_string(fixedC);
            cout << "Generating code for " << s.name << " " << B_type << "\n";

            for (unsigned k = 0; k < ukernel_shape.size(); k++) {
              printf(
                  "shape: %d x %d * 32\n",
                  ukernel_shape[k][0],
                  ukernel_shape[k][1]);

              string p1 = "GemmParams* gp";

              funcname[k] = "gemmkernel_" + to_string(ukernel_shape[k][0]) +
                  "x" + to_string(ukernel_shape[k][1]) + "_";
              funcname[k] += prefix;

              fargs = "(" + p1 + ")";

              fheader[k] =
                  "void __attribute__((noinline)) " + funcname[k] + fargs;
              srcfile << fheader[k] << " {\n";

              unsigned last_free_ymmreg = 0;
              // produce register block of C
              vector<vector<string>> vCtile(ukernel_shape[k][0]);
              for (auto r = 0; r < ukernel_shape[k][0]; r++)
                for (auto c = 0; c < ukernel_shape[k][1]; c++) {
                  vCtile[r].push_back("ymm" + to_string(last_free_ymmreg));
                  last_free_ymmreg++;
                }
              assert(last_free_ymmreg <= 14);

              string vAtmp = "ymm" + to_string(last_free_ymmreg++);
              // produce register block of B col
              vector<string> vBcol(ukernel_shape[k][1]);

              for (auto c = 0; c < ukernel_shape[k][1]; c++) {
                vBcol[c] = ("ymm" + to_string(last_free_ymmreg));
                last_free_ymmreg++;
              }

              assert(last_free_ymmreg <= 16);

              srcfile << "  asm volatile(\n";

              srcfile << "#if !defined(__clang__)"
                      << "\n";
              addi(srcfile, "mov r14, %[gp]");
              srcfile << "#else\n";
              addi(srcfile, "mov %[gp], %%r14");
              addi(srcfile, ".intel_syntax noprefix");
              srcfile << "#endif\n";

              srcfile << "\n      // Copy parameters\n";
              srcfile << "      // k\n";
              addi(srcfile, "mov r8, [r14 + 0]");
              srcfile << "      // A\n";
              addi(srcfile, "mov r9, [r14 + 8]");
              srcfile << "      // B\n";
              addi(srcfile, "mov r10, [r14 + 16]");
              srcfile << "      // beta\n";
              addi(srcfile, "mov r15, [r14 + 24]");
              srcfile << "      // accum\n";
              addi(srcfile, "mov rdx, [r14 + 32]");
              srcfile << "      // C\n";
              addi(srcfile, "mov r12, [r14 + 40]");
              srcfile << "      // ldc\n";
              addi(srcfile, "mov r13, [r14 + 48]");
              srcfile << "      // b_block_cols\n";
              addi(srcfile, "mov rdi, [r14 + 56]");
              srcfile << "      // b_block_size\n";
              addi(srcfile, "mov rsi, [r14 + 64]");
              srcfile << "      // Make copies of A and C\n";
              addi(srcfile, "mov rax, r9");
              addi(srcfile, "mov rcx, r12");
              srcfile << "\n";

              addi(srcfile, "mov rbx, 0");

              string exitlabel = "L_exit%=";
              string label2 = "loop_outter%=";
              addi(srcfile, label2 + ":");
              addi(srcfile, "mov r14, 0");

              // set all vCtile regs to zeros
              for (auto r = 0; r < vCtile.size(); r++) {
                for (auto c = 0; c < vCtile[r].size(); c++) {
                  addi(
                      srcfile,
                      "vxorps " + vCtile[r][c] + "," + vCtile[r][c] + "," +
                          vCtile[r][c]);
                }
              }

              // start marker
              if (iaca) {
                addi(srcfile, "mov ebx, 111");
                addi(srcfile, ".byte 0x64, 0x67, 0x90");
              }

              srcfile << "\n";

              srcfile << "\n";
              string label = "loop_inner%=";
              addi(srcfile, label + ":");
              srcfile << "\n";

              for (int c = 0; c < vCtile[0].size(); c++) {
                addi(
                    srcfile,
                    "vcvtph2ps " + vBcol[c] + ",XMMWORD PTR [r10 + " +
                        to_string(16 * c) + "]");
              }

              for (int r = 0; r < vCtile.size(); r++) {
                addi(
                    srcfile,
                    "vbroadcastss " + vAtmp + ",DWORD PTR [r9+" +
                        to_string(4 * r) + "]");
                for (int c = 0; c < vCtile[0].size(); c++) {
                  addi(
                      srcfile,
                      "vfmadd231ps " + vCtile[r][c] + "," + vBcol[c] + "," +
                          vAtmp);
                }
              }

              addi(
                  srcfile,
                  "add r9," + to_string(4 * ukernel_shape[k][0]),
                  fixedA); // move A ptr

              addi(
                  srcfile,
                  "add r10," + to_string(16 * ukernel_shape[k][1]),
                  fixedA); // move A ptr

              addi(srcfile, "inc r14");
              addi(srcfile, "cmp r14, r8");
              addi(srcfile, "jl " + label);

              srcfile << "\n";

              addi(srcfile, exitlabel + ":");

              // addi(srcfile, "add r10, rsi");
              srcfile << "\n";

              // end marker
              if (iaca) {
                addi(srcfile, "mov ebx, 222");
                addi(srcfile, ".byte 0x64, 0x67, 0x90");
              }

              addi(srcfile, "cmp rdx, 1");
              addi(srcfile, "je L_accum%=");
              srcfile << "      // Dump C\n";

              for (auto r = 0; r < vCtile.size(); r++) {
                for (auto c = 0; c < vCtile[r].size(); c++) {
                  addi(
                      srcfile,
                      "vmovups YMMWORD PTR [r12 + " + to_string(32 * c) +
                          "], " + vCtile[r][c],
                      fixedC);
                }
                addi(srcfile, "add r12, r13", fixedC); // move C ptr
              }
              addi(srcfile, "jmp L_done%=");

              srcfile << "\n";
              addi(srcfile, "L_accum%=:");
              srcfile << "      // Dump C with accumulate\n";

              string r_spare = (s.avx == 1) ? "ymm14" : "ymm15";
              addi(
                  srcfile,
                  "vbroadcastss " + r_spare + string(",DWORD PTR [r15]"),
                  fixedC);
              // store out C
              for (auto r = 0; r < vCtile.size(); r++) {
                for (auto c = 0; c < vCtile[r].size(); c++) {
                  switch (s.avx) {
                    case 1:
                      addi(
                          srcfile,
                          string("vmulps ymm15, ") + r_spare + comma +
                              "YMMWORD PTR [r12 + " + to_string(32 * c) + "]",
                          fixedC);
                      addi(
                          srcfile,
                          "vaddps " + vCtile[r][c] + "," + vCtile[r][c] + "," +
                              "ymm15",
                          fixedC);
                      break;
                    case 2:
                      addi(
                          srcfile,
                          "vfmadd231ps " + vCtile[r][c] + "," + r_spare + "," +
                              "YMMWORD PTR [r12 + " + to_string(32 * c) + "]",
                          fixedC);
                      break;
                    default:
                      assert(0);
                  }
                  addi(
                      srcfile,
                      "vmovups YMMWORD PTR [r12 + " + to_string(32 * c) +
                          "], " + vCtile[r][c],
                      fixedC);
                }
                addi(srcfile, "add r12, r13", fixedC); // move C ptr
              }

              srcfile << "\n";
              addi(srcfile, "L_done%=:");

              srcfile << "\n      // next outer iteration\n";
              // C
              addi(
                  srcfile,
                  "add rcx, " + to_string(32 * ukernel_shape[k][1]),
                  fixedC);
              addi(srcfile, "mov r12, rcx", fixedC);
              // A
              addi(srcfile, "mov r9, rax");

              addi(srcfile, "inc rbx");
              addi(srcfile, "cmp rbx, rdi");
              addi(srcfile, "jl " + label2);

              // output
              srcfile << "      :\n";
              // input
              srcfile << "      : [gp] \"rm\"(gp)\n";

              // clobbered
              srcfile
                  << "      : \"r8\",\n        \"r9\",\n        \"r10\",\n"
                     "        \"r11\",\n        \"r15\",\n        \"r13\",\n"
                     "        \"r14\",\n        \"rax\",\n        \"rcx\",\n"
                     "        \"rdx\",\n        \"rsi\",\n        \"rdi\",\n"
                     "        \"rbx\",\n        \"r12\",\n"
                     "        \"memory\");\n";
              srcfile << "}\n";
            }

            for (unsigned k = 0; k < ukernel_shape.size(); k++) {
              hdrfile << fheader[k] << ";\n";
            }

            fptr_typedef[B_type] =
                "typedef void (*funcptr_" + B_type + ")" + fargs;
          }
        }

  srcfile << "\n} // namespace fbgemm\n";
  srcfile.close();

  hdrfile << fptr_typedef["fp16"] << ";\n";
  hdrfile << fptr_typedef["fp32"] << ";\n";
  hdrfile << "\n} // namespace fbgemm\n\n";
  hdrfile << "#endif\n";
  hdrfile.close();
}
