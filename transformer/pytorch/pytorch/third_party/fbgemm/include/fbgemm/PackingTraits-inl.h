/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

/*
 * This file configures the important cache blocking parameters and registers
 * blocking parameters for the matrix multiplication loops inside FBGEMM.
 *
 * ROW_INTERLEAVE: the number of interleaved rows to use vpmaddubsw instructions
 * for packing B matrix. For 32-bit accumulation, ROW_INTERLEAVE = 4; For 16-bit
 * accumulation, ROW_INTERLEAVE = 2.
 *
 * VLEN: the vector length of one SIMD register. For avx2, VLEN = 256; For
 * avx512, VLEN = 512.
 *
 * NR: the register blocking parameters for N dimension. The total registers
 * used in N dimension for C accumulations are NR * ROW_INTERLEAVE * 8 (int8) /
 * VLEN.
 *
 * MR: the register blocking parameters for M dimension. The total number of
 * registers used in M dimension for C accumulations is MR.  This indicates the
 * number of vpbroadcastw instructions for A.
 *
 * (MR) * (NR * ROW_INTERLEAVE * 8 (int8) / VLEN): the number of registers used
 * for C accumulations. This number should be less than the maximum registers we
 * can use for C accumulations (A max of 12 out of 16 ymm registers for avx2; a
 * max of 28 out of 32 zmm registers for avx512 ). The remaining are used for A
 * matrix loading, B matrix loading and as temp registers. C accumulation
 * registers should be as large as possible to increase the register
 * utilization.
 *
 * MCB: the cache blocking parameters for M dimension. MCB needs to be a
 * multiple of MR.
 *
 * NCB: the cache blocking parameters for N dimension. NCB needs to be a
 * multiple of NR.
 *
 * KCB: the cache blocking parameters for K dimension. KCB needs to be a
 * multiple of ROW_INTERLEAVE.
 */

/**
 * @brief Packing parameter specialization for accumulation into 32-bit
 * integers.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx2.
 */
template <typename T>
struct PackingTraits<
    T,
    std::int32_t,
    inst_set_t::avx2,
    typename std::enable_if<is_8bit<T>::value>::type> {
  static constexpr int MR{12}; ///< Register block for M dimension.
  static constexpr int NR{8}; ///< Register block for N dimension.
                              ///< NR = VLEN/8/ROW_INTERLEAVE = 256 / 8 / 4 = 8.
                              ///< Total registers used for N dimension: NCB/NR.
                              ///< Here we use 12 x 1 ymm register blocking for
                              ///< the registers used for accumulation C.

  static constexpr int ROW_INTERLEAVE{
      4}; ///< 4 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      120}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      8}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{512}; ///< Cache block for K dimension.
};

/**
 * @brief Packing parameter specialization for accumulation into 16-bit
 * integers.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx2.
 */
template <typename T>
struct PackingTraits<
    T,
    std::int16_t,
    inst_set_t::avx2,
    typename std::enable_if<is_8bit<T>::value>::type> {
  static constexpr int MR{3}; ///< Register block for M dimension.
  static constexpr int NR{
      16}; ///< Register block for N dimension;
           ///< NR = VLEN/8/ROW_INTERLEAVE = 256 / 8 / 2 = 16.
           ///< Total registers used for N dimension: NCB/NR.
           ///< Here we use 3 x 4 ymm register blocking for the
           ///< registers used for accumulation C.

  static constexpr int ROW_INTERLEAVE{
      2}; ///< 2 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      60}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      64}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{256}; ///< Cache block for K dimension.
};

/**
 * @brief Packing parameter specialization for float input and float
 * accumulation.
 *
 * This is picked when template paramtere T is of float type and instruction
 * set is avx2.
 */
template <>
struct PackingTraits<float, float, inst_set_t::avx2> {
  static constexpr int MR{3}; ///< Register block for M dimension
  static constexpr int NR{32}; ///< Register block for N dimension

  static constexpr int ROW_INTERLEAVE{1}; ///< No Row interleave.

  static constexpr int MCB{
      24}; ///< Cache block for M dimension (multiple of MR)
  static constexpr int NCB{
      64}; ///< Cache block for N dimension (multiple of NR)
  static constexpr int KCB{256}; ///< Cache block for K dimension
};

/**
 * @brief Packing parameter specialization for fp16 input and float
 * accumulation.
 *
 * This is picked when template parameter T is of float16 type and instruction
 * set is avx2.
 */
template <>
struct PackingTraits<float16, float, inst_set_t::avx2> {
  static constexpr int BCOL{8};
  static constexpr int ROW_INTERLEAVE{1};
};

/**
 * @brief Packing parameter specialization for accumulation into 32-bit
 * integers.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx512.
 */
template <typename T>
struct PackingTraits<
    T,
    std::int32_t,
    inst_set_t::avx512,
    typename std::enable_if<is_8bit<T>::value>::type> {
  static constexpr int MR{14}; ///< Register block for M dimension.
  static constexpr int NR_MIN{
      16}; ///< Minimum register block for N dimension.
           ///< 16 because 16*ROW_INTERLEAVE int8 elements
           ///< completely fill a 512-bit wide vector.
  static constexpr int NR{
      32}; ///< Register block for N dimension.
           ///< Must be a multiple of 16 because 16*ROW_INTERLEAVE int8 elements
           ///< completely fill a 512-bit wide vector. Total registers used for
           ///< N dimension: NR*ROW_INTERLEAVE*8/VLEN. We use MR x
           ///< NR*ROW_INTERLEAVE*8/VLEN zmm registers
           ///< for C accumulations.

  static constexpr int ROW_INTERLEAVE{
      4}; ///< 4 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      56}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      32}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{256}; ///< Cache block for K dimension.
};

/**
 * @brief Packing parameter specialization for accumulation into 16-bit
 * integers.
 *
 * This is picked when T is of int8 type (signed or unsigned) and instruction
 * set is avx512.
 */
template <typename T>
struct PackingTraits<
    T,
    std::int16_t,
    inst_set_t::avx512,
    typename std::enable_if<is_8bit<T>::value>::type> {
  static constexpr int MR{6}; ///< Register block for M dimension
  static constexpr int NR_MIN{
      32}; ///< Minimum register block for N dimension;
           ///< 32 because 32*ROW_INTERLEAVE int8 elements
           ///< completely fill a 512-bit wide vector.
  static constexpr int NR{
      128}; ///< Register block for N dimension;
            ///< Must be a multiple of 32 because 32*ROW_INTERLEAVE int8
            ///< elements completely fill a 512-bit wide vector. Total registers
            ///< used for N dimension: NR*ROW_INTERLEAVE*8/VLEN. We use MR x
            ///< NR*ROW_INTERLEAVE*8/VLEN zmm registers
            ///< for C accumulations.

  static constexpr int ROW_INTERLEAVE{
      2}; ///< 2 rows are interleaved to use vpmaddubsw instruction for packing
          ///< B matrix.

  static constexpr int MCB{
      60}; ///< Cache block for M dimension (multiple of MR).
  static constexpr int NCB{
      128}; ///< Cache block for N dimension (multiple of NR).
  static constexpr int KCB{256}; ///< Cache block for K dimension.
};
