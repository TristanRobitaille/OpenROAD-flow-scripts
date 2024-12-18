"""
This class provides utility functions to rewrite the source code using Pyslang.
"""

import pyslang
from os import remove
from shutil import copy2
from pathlib import Path
from copy import deepcopy

"""
TODO
- Figure out why I get encoding error when reading the define text if using self.module instead of local module.
"""

class VerilogRewriter():
    def __init__(self, top_fp:str, pkg_fp:str):
        self.top_fp = Path(top_fp)
        self.pkg_fp = Path(pkg_fp)
        self.top_char_offset_from_insertion = 0
        self.pkg_char_offset_from_insertion = 0
        self.top_backup_path = self.top_fp.with_name(f"{self.top_fp.stem}_ref{self.top_fp.suffix}.temp")
        self.pkg_backup_path = self.pkg_fp.with_name(f"{self.pkg_fp.stem}_ref{self.pkg_fp.suffix}.temp")
        copy2(self.top_fp, self.top_backup_path)
        copy2(self.pkg_fp, self.pkg_backup_path)

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
        
        with open(fp, 'r') as file:
            content = file.read()
        new_content = content[:start_pos] + new_text + content[end_pos:]
        with open(f"{fp}", 'w') as file:
            file.write(new_content)

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
        if is_for_pkg:
            tree = pyslang.SyntaxTree.fromFile(str(self.pkg_fp), source_manager, options)
        else:
            tree = pyslang.SyntaxTree.fromFile(str(self.top_fp), source_manager, options)
        module = tree.root.members[0]

        def visit_and_update(node):
            if (node.kind == pyslang.SyntaxKind.ParameterDeclaration): # Parameter
                if node.declarators[0].name.valueText in params_left.keys():
                    start = node.declarators[0].getLastToken().range.start.offset + self.top_char_offset_from_insertion
                    end = node.declarators[0].getLastToken().range.end.offset + self.top_char_offset_from_insertion
                    self.top_char_offset_from_insertion += self._replace_in_file(start, end, str(params_left[node.declarators[0].name.valueText]), self.top_fp)
                    del params_left[node.declarators[0].name.valueText]

            elif isinstance(node, pyslang.Token): # Define
                for trivia in node.trivia:
                    if trivia.kind == pyslang.TriviaKind.Directive and trivia.syntax().kind == pyslang.SyntaxKind.DefineDirective and trivia.syntax().name.valueText in defines_left.keys():
                        start = trivia.syntax().getLastToken().range.start.offset + self.top_char_offset_from_insertion
                        end = trivia.syntax().getLastToken().range.end.offset + self.top_char_offset_from_insertion
                        self.top_char_offset_from_insertion += self._replace_in_file(start, end, str(defines_left[trivia.syntax().name.valueText]), self.top_fp)
                        del defines_left[trivia.syntax().name.valueText]

            if not defines_left and not params_left:
                return pyslang.VisitAction.Interrupt
            return pyslang.VisitAction.Advance

        module.visit(visit_and_update)

        if defines_left:
            if is_for_pkg:
                print(f"Warning: Did not find all defines to replace in package file: {defines_left}")
            else:
                print(f"Warning: Did not find all defines to replace in top-level file: {defines_left}.")
        if params_left:
            if is_for_pkg:
                print(f"Warning: Did not find all parameters to replace in package file: {params_left}")
            else:
                print(f"Warning: Did not find all parameters to replace in top-level file: {params_left}.")

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

        self._update_file(top_new_def, top_new_params, is_for_pkg=False)
        self._update_file(pkg_new_def, pkg_new_params, is_for_pkg=True)

    def reset(self) -> None:
        """
        Reset the top-level and package files to their original state.

        Parameters: None

        Returns: None
        """

        copy2(self.top_backup_path, self.top_fp)
        remove(self.top_backup_path)
        copy2(self.pkg_backup_path, self.pkg_fp)
        remove(self.pkg_backup_path)
