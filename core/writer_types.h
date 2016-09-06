// *****************************************************************************
// Filename:    writer_types.h
// Date:        2012-12-26 10:55
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef WRITER_TYPES_H_
#define WRITER_TYPES_H_

#include <cstdlib>
#include <iostream>
#include <string>

using std::cout;
using std::cerr;
using std::endl;
using std::string;

class WriterType {
 public:

  static const WriterType kConsoleWriter;
  static const WriterType kDummyWriter;
  static const WriterType kMultipleFileWriter;
  static const WriterType kSingleFileWriter;
  static const WriterType kConsoleTestWriter;
  static const WriterType kFileTestWriter;

  WriterType() : type(CONSOLE_WRITER) {
  }

  explicit WriterType(const WriterType &rhs) : type(rhs.type) {
  }

  static const WriterType& GetWriterTypeFromString(const string &description) {
    if (description == "console") {
      return kConsoleWriter;
    } else if (description == "single_file") {
      return kSingleFileWriter;
    } else if (description == "multiple_file") {
      return kMultipleFileWriter;
    } else if (description == "console_test") {
      return kConsoleTestWriter;
    } else if (description == "file_test") {
      return kFileTestWriter;
    } else if (description == "dummy") {
      return kDummyWriter;
    } else {
      cout << "WriterType::GetWriterTypeFromString error: invalid writer type!"
           << endl;
      exit(1);
    }
  }

  WriterType& operator=(const WriterType &rhs) {
    type = rhs.type;
    return *this;
  }

  bool operator==(const WriterType &rhs) const {
    return type == rhs.type;
  }

  string GetDescriptionString() const {
    switch (type) {
      case CONSOLE_WRITER:
        return "ConsoleWriter";
      case DUMMY_WRITER:
        return "DummyWriter";
      case MULTIPLE_FILE_WRITER:
        return "MultipleFileWriter";
      case SINGLE_FILE_WRITER:
        return "SingleFileWriter";
      case CONSOLE_TEST_WRITER:
        return "ConsoleTestWriter";
      case FILE_TEST_WRITER:
        return "FileTestWriter";
      default:
        cout << "WriterType::GetDescriptionString error: invalid writer type!"
             << endl;
        exit(1);
    }
  }

 private:

  enum WriterTypeEnum {
    CONSOLE_WRITER,
    DUMMY_WRITER,
    MULTIPLE_FILE_WRITER,
    SINGLE_FILE_WRITER,
    CONSOLE_TEST_WRITER,
    FILE_TEST_WRITER,
  };

  explicit WriterType(const WriterTypeEnum &val) : type(val) {
  }

  WriterTypeEnum type;
};

#endif
