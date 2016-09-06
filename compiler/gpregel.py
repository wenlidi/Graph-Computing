#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys, os, getopt, re

# map['io']   = 'in' or 'out' or ''
# map['type'] = 'int' or 'unsigned int' or 'char' or 'bool' or 'float'
# map['name'] = user-defined-member-name
# map['val''] = 'RandUtil::'FuncName or user defined compile-time constants if map['io'] is 'in',
#               user defined compile-time constants if map['io'] is 'out',
#               '' otherwise
gmap = []
vmap = []
emap = []
mmap = []

def GetMappingForOneStatement(name, match):
  str_map = {}
  str_map['io'] = 'in' if name == 'Global' else match[0][0]
  str_map['type'] = match[0][1]
  str_map['name'] = match[0][2]

  val = match[0][3]
  if val.__len__() == 0 or val.isspace():
    val = '0'
  else:
    if str_map['io'] == 'in':
      if val == 'RAND_VERTEX_ID':
        val = 'RandUtil::RandVertexId()'
      elif val == 'RAND_SMALL_UINT':
        val = 'RandUtil::RandSmallUInt()'
      elif val == 'RAND_MIDDLE_UINT':
        val = 'RandUtil::RandMiddleUInt()'
      elif val == 'RAND_LARGE_UINT':
        val = 'RandUtil::RandLargeUInt()'
      elif val == 'RAND_FLOAT':
        val = 'RandUtil::RandFloat()'
      elif val == 'RAND_FRACTION':
        val = 'RandUtil::RandFraction()'
    elif str_map['io'] == '':
      print('Error: initial value is not applicable with empty io type.')
      sys.exit()

  str_map['val'] = val
  return str_map

def FindDefinition(name, content, should_have_io):
  str_io_qualifier = r'(?:in|out)\s+' if should_have_io else ''
  str_statement = str_io_qualifier + r'(?:int|unsigned\s*int|char|bool|float)\s+[_a-zA-Z]\w*\s*(?:=[^;]+)?;'
  str_valid_block_definition = r'\{(?:\s*' + str_statement + r')*\s*\}'
  str_struct = r'(?<!\w)struct\s+' + name + r'\s+' + str_valid_block_definition
  print('struct pattern: ', str_struct)

  pat_struct = re.compile(str_struct)
  def_struct = re.findall(pat_struct, content)
  if def_struct.__len__() != 1:
    print('Could not find or find multiple definition of ' + name + ', or the definition has syntax error!')
    sys.exit()
  print(def_struct)

  pat_statement = re.compile(str_statement)
  def_member = re.findall(pat_statement, def_struct[0])
  print('statement pattern: ', str_statement)
  print(def_member)

  match_str_io_qualifier = r'(in|out)\s+' if should_have_io else '()'
  match_str_statement = match_str_io_qualifier + r'(int|unsigned\s*int|char|bool|float)\s+([_a-zA-Z]\w*)\s*(?:=\s*([^;]+))?;'
  pat_match_statement = re.compile(match_str_statement)
  print('matched statement pattern: ', match_str_statement)

  result = []
  for mem_str in def_member:
    match = re.findall(pat_match_statement, mem_str)
    print(match)
    result.append(GetMappingForOneStatement(name, match))

  print(result)
  print()
  return result

def ParseDataTypes(data_type_file):
  global gmap, vmap, emap, mmap
  f = open(data_type_file, 'r')
  content = f.read()
  f.close()
  gmap = FindDefinition('Global', content, 0)
  vmap = FindDefinition('Vertex', content, 1)
  emap = FindDefinition('Edge', content, 1)
  mmap = FindDefinition('Message', content, 0)

def GetTranslatedContent(matched_list):
  global gmap, vmap, emap, mmap
  map_type = matched_list[0][0]
  content = matched_list[0][1]
  member_map = []
  io_type = ''

  if map_type == 'G':
    member_map = gmap
    io_type = 'in'
  elif map_type == 'V' or map_type == 'E':
    io_type = 'io'
    member_map = vmap if map_type == 'V' else emap
  elif map_type == 'V_IN' or map_type == 'E_IN':
    io_type = 'in'
    member_map = vmap if map_type == 'V_IN' else emap
  elif map_type == 'V_OUT' or map_type == 'E_OUT':
    io_type = 'out'
    member_map = vmap if map_type == 'V_OUT' else emap
  else:
    member_map = mmap
    io_type = ''

  result = []
  for m in member_map:
    if io_type == 'io' or m['io'] == io_type:
      translated = content.replace('<GP_TYPE>', m['type'])
      translated = translated.replace('<GP_NAME>', m['name'])
      if m['io'] == 'in':
        translated = translated.replace('<GP_RAND_VALUE>', m['val'])
      else:
        translated = translated.replace('<GP_INIT_VALUE>', m['val'])
      result.append(translated)
  return result

def GenerateOutFile(template_file, output_dir):
  global gmap, vmap, emap, mmap
  filename = os.path.split(template_file)[-1]
  full_output_name = os.path.join(output_dir, filename)
  print('Output to: ', full_output_name)

  f = open(template_file, 'r')
  lines = f.readlines()
  f.close()

  pattern = re.compile(r'\s*\$\$(\w+)\[\[(.+)\]\]')
  f = open(full_output_name, 'w')
  for l in lines:
    m = re.findall(pattern, l)
    if m.__len__() != 0:
      translated = GetTranslatedContent(m)
      print('--- ', m)
      for t in translated:
        f.write(t)
        f.write('\n')
        print('------ ', t)
    else:
      f.write(l)
  f.close()
  print()

def Compile(data_type_file, templates, output_dir):
  ParseDataTypes(data_type_file)
  for t in templates:
    GenerateOutFile(t, output_dir)
  
def Usage():
  print("Usage: gpregel -d data_type_file -t template_file1[[,template_file2]...] -o output_directory")

if __name__ == "__main__":
  opts, args = getopt.getopt(sys.argv[1:], "hd:t:o:")
  data_type_file = ''
  templates = ''
  output_dir = ''
  found_param = 0

  for op, value in opts:
    if op == "-d":
      data_type_file = value
      found_param += 1
    elif op == "-t":
      templates = value.split(',')
      found_param += 1
    elif op == "-o":
      output_dir = value
      found_param += 1
    elif op == "-h":
      Usage()
      sys.exit()
  if found_param != 3:
    Usage()
    sys.exit()

  print("data type file: ", data_type_file)
  print("template files: ", templates)
  print("output directory: ", output_dir)
  Compile(data_type_file, templates, output_dir)
