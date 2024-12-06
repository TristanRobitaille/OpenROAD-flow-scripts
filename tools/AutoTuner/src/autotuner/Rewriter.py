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
    def __init__(self, sv_fp:str):
        self.sv_fp = Path(sv_fp)
        self.char_offset_from_insertion = 0
        self.backup_path = self.sv_fp.with_name(f"{self.sv_fp.stem}_ref{self.sv_fp.suffix}")
        copy2(self.sv_fp, self.backup_path)

    def _replace_in_file(self, start_pos:int, end_pos:int, new_text:str) -> int:
        """
        Replace the text in the file at the given position and return the string length delta between the new text and the old text.

        Parameters:
        -start_pos (int): The starting index of the text to be replaced.
        -end_pos (int): The ending index of the text to be replaced.
        -new_text (str): The new text to place starting at start_pos.

        Return:
        -int: Difference in length between the new text and the old text.
        """
        
        with open(self.sv_fp, 'r') as file:
            content = file.read()
        new_content = content[:start_pos] + new_text + content[end_pos:]
        with open(f"{self.sv_fp}", 'w') as file:
            file.write(new_content)

        return len(new_text) - (end_pos - start_pos)

    def update_sv(self, new_def: dict, new_params: dict) -> None:
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
        tree = pyslang.SyntaxTree.fromFile(str(self.sv_fp), source_manager, options)
        module = tree.root.members[0]

        def visit_and_update(node):
            if (node.kind == pyslang.SyntaxKind.ParameterDeclaration): # Parameter
                if node.declarators[0].name.valueText in params_left.keys():
                    start = node.declarators[0].getLastToken().range.start.offset + self.char_offset_from_insertion
                    end = node.declarators[0].getLastToken().range.end.offset + self.char_offset_from_insertion
                    self.char_offset_from_insertion += self._replace_in_file(start, end, str(params_left[node.declarators[0].name.valueText]))
                    del params_left[node.declarators[0].name.valueText]

            elif isinstance(node, pyslang.Token): # Define
                for trivia in node.trivia:
                    if trivia.kind == pyslang.TriviaKind.Directive and trivia.syntax().kind == pyslang.SyntaxKind.DefineDirective and trivia.syntax().name.valueText in defines_left.keys():
                        start = trivia.syntax().getLastToken().range.start.offset + self.char_offset_from_insertion
                        end = trivia.syntax().getLastToken().range.end.offset + self.char_offset_from_insertion
                        self.char_offset_from_insertion += self._replace_in_file(start, end, str(defines_left[trivia.syntax().name.valueText]))
                        del defines_left[trivia.syntax().name.valueText]

            if not defines_left and not params_left:
                return pyslang.VisitAction.Interrupt
            return pyslang.VisitAction.Advance

        module.visit(visit_and_update)

        if defines_left:
            print(f"Warning: Did not find all defines to replace: {defines_left}")
        if params_left:
            print(f"Warning: Did not find all parameters to replace: {params_left}")

    def reset_sv(self) -> None:
        """
        Reset the SystemVerilog file to its original state.

        Parameters: None

        Returns: None
        """

        copy2(self.backup_path, self.sv_fp)
        remove(self.backup_path)
