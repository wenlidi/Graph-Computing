// *****************************************************************************
// Filename:    macro.h
// Date:        2012-12-11 13:41
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef MACRO_H_
#define MACRO_H_

#define DISABLE_EVIL_DEFAULT_FUNCTIONS(CLASS) \
    CLASS(const CLASS &other); \
    CLASS& operator=(const CLASS &other)

#endif
