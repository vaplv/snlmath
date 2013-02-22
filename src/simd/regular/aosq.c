/*
 * Copyright (c) 2013 Vincent Forest
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO
 * EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "simd/aosq.h"
#include "simd/aosf33.h"
#include <sys/sys.h>

vf4_t
aosq_slerp(vf4_t from, vf4_t to, vf4_t vvvv)
{
  const float vf = vf4_x(vvvv);
  if(vf == 0.f)
    return from;
  else if(vf == 1.f)
    return to;

  const vf4_t tmp_cos_omega = vf4_dot(from, to);

  const vf4_t mask0 = vf4_lt(tmp_cos_omega, vf4_zero());
  const vf4_t tmp0 = vf4_sel(to, vf4_minus(to), mask0);
  const vf4_t cos_omega =
  vf4_sel(tmp_cos_omega, vf4_minus(tmp_cos_omega), mask0);

  const vf4_t omega = vf4_acos(cos_omega);
  const vf4_t rcp_sin_omega = vf4_rcp(vf4_sin(omega));
  const vf4_t one_sub_v = vf4_sub(vf4_set1(1.f), vvvv);
  const vf4_t tmp1 = vf4_mul(vf4_sin(vf4_mul(one_sub_v, omega)), rcp_sin_omega);
  const vf4_t tmp2 = vf4_mul(vf4_sin(vf4_mul(omega, vvvv)), rcp_sin_omega);

  const vf4_t mask1 =
  vf4_gt(vf4_sub(vf4_set1(1.f), cos_omega), vf4_set1(1.e-6f));
  const vf4_t scale0 = vf4_sel(one_sub_v, tmp1, mask1);
  const vf4_t scale1 = vf4_sel(vvvv, tmp2, mask1);

  return vf4_madd(from, scale0, vf4_mul(tmp0, scale1));
}

void
aosq_to_aosf33(vf4_t q, struct aosf33* out)
{
  const vf4_t i2j2k2_ = vf4_add(q, q);

  const vf4_t r0 = /* { jj2 + kk2, ij2 + ak2, ik2 - aj2 } */
    vf4_madd(vf4_mul(vf4_zzyy(i2j2k2_), vf4_zwwz(q)),
             vf4_set(1.f, 1.f, -1.f, 0.f),
             vf4_mul(vf4_yyzz(i2j2k2_), vf4_yxxy(q)));
  const vf4_t r1 = /* { ij2 - ak2, ii2 + kk2, jk2 + ai2 } */
    vf4_madd(vf4_mul(vf4_zzxx(i2j2k2_), vf4_wzwz(q)),
             vf4_set(-1.f, 1.f, 1.f, 0.f),
             vf4_mul(vf4_yxzw(i2j2k2_), vf4_xxyy(q)));
  const vf4_t r2 = /* { ik2 + aj2, jk2 - ai2, ii2 + jj2 } */
    vf4_madd(vf4_mul(vf4_yxyx(i2j2k2_), vf4_wwyy(q)),
             vf4_set(1.f, -1.f, 1.f, 0.f),
             vf4_mul(vf4_zzxx(i2j2k2_), vf4_xyxy(q)));

  out->c0 = /* { 1 - (jj2 + kk2), ij2 + ak2, ik2 - aj2 } */
    vf4_madd(r0, vf4_set(-1.f, 1.f, 1.f, 0.f), vf4_set(1.f, 0.f, 0.f, 0.f));
  out->c1 = /* { ij2 - ak2, 1 - (ii2 + kk2), jk2 + ai2 } */
    vf4_madd(r1, vf4_set(1.f, -1.f, 1.f, 0.f), vf4_set(0.f, 1.f, 0.f, 0.f));
  out->c2 = /* { ik2 + aj2, jk2 - ai2, 1 - (ii2 + jj2) } */
    vf4_madd(r2, vf4_set(1.f, 1.f, -1.f, 0.f), vf4_set(0.f, 0.f, 1.f, 0.f));
}

