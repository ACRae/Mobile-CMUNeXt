import re


class LatexHelper:
    @staticmethod
    def normalize_latexstring(string):
        # Replace common LaTeX special characters with their escaped versions
        string = string.replace("\\", "\\textbackslash")  # Escape backslashes
        string = string.replace("_", "\\_")  # Escape underscores
        string = string.replace("%", "\\%")  # Escape percentage signs
        string = string.replace("$", "\\$")  # Escape dollar signs
        string = string.replace("&", "\\&")  # Escape ampersands
        string = string.replace("#", "\\#")  # Escape hash marks
        string = string.replace("{", "\\{")  # Escape opening curly braces
        string = string.replace("}", "\\}")  # Escape closing curly braces
        string = string.replace("~", "\\~")  # Escape tilde
        string = string.replace("^", "\\^")  # Escape caret
        string = string.replace("'", "\\textquotesingle")  # Escape single quotes
        string = string.replace('"', "\\textquotedbl")  # Escape double quotes

        # Optionally, you can also replace any other character that may interfere with LaTeX
        # For example, remove or replace other non-ASCII characters
        string = re.sub(r"[^\x00-\x7F]+", "", string)  # Remove non-ASCII characters

        return string

    @staticmethod
    def write_entries(data: list, comment: str):
        # Join each inner list, adding '& ' to all but the last item
        joined_data = " ".join(["& " + str(item) for item in data])
        return " ".join([joined_data, "\t% " + comment])

    class Tabular:
        @staticmethod
        def open_tabular(n_cols: int) -> str:
            """ """
            COLUMN = "c"
            table_str = "\n\\begin{tabular} {"
            for idx in range(n_cols):
                table_str += COLUMN
                if idx != n_cols - 1:
                    table_str += "|"
            table_str += "}\n\\toprule[1.1pt]\n\n"
            table_str += f"% {n_cols} COLUMNS\n\n"
            table_str += "% <-- LABELS HERE --> \n\n"
            table_str += "\\midrule[0.7pt]\n\n"
            return table_str

        @staticmethod
        def close_tabular() -> str:
            return "\n\\bottomrule[1.1pt]\n\\end{tabular}"
