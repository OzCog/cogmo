/*
 * opencog/util/cogutil.h
 *
 * Copyright (C) 2015 by OpenCog Foundation
 * All Rights Reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program; if not, write to:
 * Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _OPENCOG_COGUTIL_H
#define _OPENCOG_COGUTIL_H

#define COGUTIL_MAJOR_VERSION 2
#define COGUTIL_MINOR_VERSION 0
#define COGUTIL_MICRO_VERSION 3

#define COGUTIL_VERSION_STRING "2.0.3"

// Foundation Layer: Tensor Architecture Configuration
// Cognitive Function: utility-primitives
// Tensor Shape: [512, 128, 8] = 524,288 DOF
#define COGUTIL_TENSOR_SHAPE_X 512
#define COGUTIL_TENSOR_SHAPE_Y 128
#define COGUTIL_TENSOR_SHAPE_Z 8
#define COGUTIL_DEGREES_OF_FREEDOM (COGUTIL_TENSOR_SHAPE_X * COGUTIL_TENSOR_SHAPE_Y * COGUTIL_TENSOR_SHAPE_Z)

// Compile-time tensor validation
static_assert(COGUTIL_DEGREES_OF_FREEDOM == 524288, 
              "Cogutil tensor DOF must equal 524,288 as per Foundation Layer specification");

// Tensor type definitions for cognitive operations
namespace opencog { namespace util {
    
    // Foundation Layer tensor types
    struct TensorShape3D {
        static constexpr size_t X = COGUTIL_TENSOR_SHAPE_X;
        static constexpr size_t Y = COGUTIL_TENSOR_SHAPE_Y; 
        static constexpr size_t Z = COGUTIL_TENSOR_SHAPE_Z;
        static constexpr size_t DOF = COGUTIL_DEGREES_OF_FREEDOM;
    };
    
    // Cognitive function identifier
    static constexpr const char* COGNITIVE_FUNCTION = "utility-primitives";
    
}} // namespace opencog::util

#endif // _OPENCOG_COGUTIL_H
