"""
Author: Tristan Robitaille, ETH Zürich 2024

This class provides utility functions to overwrite parameters and defines in SystemVerilog source code using Pyslang.

Usage:
1. Initialize the VerilogRewriter with the paths to the top-level and package files.
2. Use the update_sv method to update define directives and parameters in both files.

Dependencies:
- pyslang: For parsing and manipulating SystemVerilog syntax trees.
- os: For file removal operations.
- shutil: For file copying operations.
- pathlib: For handling file paths.
- copy: For deep copying dictionaries.
- enum: For defining terminal tool messages.

TODO:
- Figure out why I get encoding error when reading the define text if using self.module instead of local module.
"""

import pyslang
from os import remove
from enum import Enum
from shutil import copy2
from pathlib import Path
from copy import deepcopy

# ----- ENUMS ----- #
class TerminalTool(Enum):
    warning = "\033[93mVERILOG REWRITER WARNING\033[0m"
    error = "\033[91mVERILOG REWRITER ERROR\033[0m"

# ----- CLASSES ----- #
class VerilogRewriter():
    def __init__(self, top_fp: str = None, pkg_fp: str = None):
        self.top_fp = None
        self.pkg_fp = None
        if top_fp is not None:
            self.top_fp = Path(top_fp)
        if pkg_fp is not None:
            self.pkg_fp = Path(pkg_fp)
        self.top_char_offset_from_insertion = 0
        self.pkg_char_offset_from_insertion = 0

    def _replace_in_file(self, start_pos:int, end_pos:int, new_text:str, fp:str) -> int:
        """
        Replace the text in the file at the given position and return the string length delta between the new text and the old text.

        Parameters:
        -start_pos (int): The starting index of the text to be replaced.
        -end_pos (int): The ending index of the text to be replaced.
        -new_text (str): The new text to place starting at start_pos.

        Return:
        -int: Difference in length between the new text and the old text.
        """

        try:
            with open(fp, 'r') as file:
                content = file.read()
        except FileNotFoundError:
            print(f"{TerminalTool.error.value}: File {fp} not found.")
            return 0
        except IOError as e:
            print(f"{TerminalTool.error.value}: IOError reading file {fp}. {e}")
            return 0

        new_content = content[:start_pos] + new_text + content[end_pos:]

        try:
            with open(fp, 'w') as file:
                file.write(new_content)
        except IOError as e:
            print(f"{TerminalTool.error.value}: IOError writing to file {fp}. {e}")
            return 0

        return len(new_text) - (end_pos - start_pos)

    def _update_file(self, new_def: dict, new_params: dict, is_for_pkg: bool) -> None:
        """
        Update given define directive in the SystemVerilog file.

        Parameters:
        -new_def (dict): Dictionary containing the define values to change.
        -new_params (dict): Dictionary containing the parameter values to change.

        Returns: None
        """

        defines_left = deepcopy(new_def)
        params_left = deepcopy(new_params)

        source_manager = pyslang.SourceManager()
        preprocessor_options = pyslang.PreprocessorOptions()
        options = pyslang.Bag([preprocessor_options])
        try:
            if is_for_pkg:
                tree = pyslang.SyntaxTree.fromFile(str(self.pkg_fp), source_manager, options)
            else:
                tree = pyslang.SyntaxTree.fromFile(str(self.top_fp), source_manager, options)
        except Exception as e:
            print(f"{TerminalTool.error.value}: Exception creating syntax tree. {e}")
            return
        module = tree.root.members[0]

        def visit_and_update(node):
            if (node.kind == pyslang.SyntaxKind.ParameterDeclaration): # Parameters
                if node.declarators[0].name.valueText in params_left.keys():
                    start = node.declarators[0].getLastToken().range.start.offset + self.top_char_offset_from_insertion
                    end = node.declarators[0].getLastToken().range.end.offset + self.top_char_offset_from_insertion
                    if is_for_pkg:
                        self.top_char_offset_from_insertion += self._replace_in_file(start, end, str(params_left[node.declarators[0].name.valueText]), self.pkg_fp)
                    else:
                        self.top_char_offset_from_insertion += self._replace_in_file(start, end, str(params_left[node.declarators[0].name.valueText]), self.top_fp)
                    del params_left[node.declarators[0].name.valueText]

            elif isinstance(node, pyslang.Token): # Defines
                for trivia in node.trivia:
                    if trivia.kind == pyslang.TriviaKind.Directive and trivia.syntax().kind == pyslang.SyntaxKind.DefineDirective and trivia.syntax().name.valueText in defines_left.keys():
                        start = trivia.syntax().getLastToken().range.start.offset + self.top_char_offset_from_insertion
                        end = trivia.syntax().getLastToken().range.end.offset + self.top_char_offset_from_insertion
                        if is_for_pkg:
                            self.top_char_offset_from_insertion += self._replace_in_file(start, end, str(defines_left[trivia.syntax().name.valueText]), self.pkg_fp)
                        else:
                            self.top_char_offset_from_insertion += self._replace_in_file(start, end, str(defines_left[trivia.syntax().name.valueText]), self.top_fp)
                        del defines_left[trivia.syntax().name.valueText]

            if not defines_left and not params_left:
                return pyslang.VisitAction.Interrupt
            return pyslang.VisitAction.Advance

        module.visit(visit_and_update)

        if defines_left:
            if is_for_pkg:
                print(f"{TerminalTool.warning.value}: Did not find all defines to replace in package file: {defines_left}")
            else:
                print(f"{TerminalTool.warning.value}: Did not find all defines to replace in top-level file: {defines_left}.")
        if params_left:
            if is_for_pkg:
                print(f"{TerminalTool.warning.value}: Did not find all parameters to replace in package file: {params_left}")
            else:
                print(f"{TerminalTool.warning.value}: Did not find all parameters to replace in top-level file: {params_left}.")

    def update_sv(self, top_new_def: dict, top_new_params: dict, pkg_new_def: dict, pkg_new_params: dict):
        """
        Update given define directives and parameters for both the top-level and package file.

        Parameters:
        -top_new_def (dict): Dictionary containing the define values to change in the top-level file.
        -top_new_params (dict): Dictionary containing the parameter values to change in the top-level file.
        -pkg_new_def (dict): Dictionary containing the define values to change in the package file.
        -pkg_new_params (dict): Dictionary containing the parameter values to change in the package file.

        Returns: None
        """

        if self.top_fp is not None:
            self._update_file(top_new_def, top_new_params, is_for_pkg=False)
        if self.pkg_fp is not None:
            self._update_file(pkg_new_def, pkg_new_params, is_for_pkg=True)
