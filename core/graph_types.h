// *****************************************************************************
// Filename:    graph_types.h
// Date:        2012-12-10 19:52
// Author:      Guangda Lai
// Email:       lambda2fei@gmail.com
// Description: TODO(laigd): Put the file description here.
// *****************************************************************************

#ifndef GRAPH_TYPES_H_
#define GRAPH_TYPES_H_

#include <cstdlib>
#include <iostream>
#include <string>

using std::cout;
using std::cerr;
using std::endl;
using std::string;

class GraphType {
 public:

  static const GraphType kGraphFromConsole;
  static const GraphType kGraphFromFile;
  static const GraphType kSimpleGraph;
  static const GraphType kRMatGraph;
  static const GraphType kRandGraph;

  GraphType() : type(GRAPH_FROM_FILE) {
  }

  explicit GraphType(const GraphType &rhs) : type(rhs.type) {
  }

  static const GraphType& GetGraphTypeFromString(const string &description) {
    if (description == "console") {
      return kGraphFromConsole;
    } else if (description == "file") {
      return kGraphFromFile;
    } else if (description == "simple") {
      return kSimpleGraph;
    } else if (description == "rmat") {
      return kRMatGraph;
    } else if (description == "rand") {
      return kRandGraph;
    } else {
      cout << "GraphType::GetGraphTypeFromString error: invalid graph type!"
           << endl;
      exit(1);
    }
  }

  GraphType& operator=(const GraphType &rhs) {
    type = rhs.type;
    return *this;
  }

  bool operator==(const GraphType &rhs) const {
    return type == rhs.type;
  }

  string GetDescriptionString() const {
    switch (type) {
      case GRAPH_FROM_CONSOLE:
        return "GraphFromConsole";
      case GRAPH_FROM_FILE:
        return "GraphFromFile";
      case GRAPH_SIMPLE:
        return "SimpleGraph";
      case GRAPH_RMAT:
        return "R-MAT Graph";
      case GRAPH_RAND:
        return "RandGraph";
      default:
        cout << "GraphType::GetDescriptionString error: invalid graph type!"
             << endl;
        exit(1);
    }
  }

 private:

  enum GraphTypeEnum {
    GRAPH_FROM_CONSOLE,
    GRAPH_FROM_FILE,
    GRAPH_SIMPLE,
    GRAPH_RMAT,
    GRAPH_RAND,
  };

  explicit GraphType(const GraphTypeEnum &val) : type(val) {
  }

  GraphTypeEnum type;
};

#endif
