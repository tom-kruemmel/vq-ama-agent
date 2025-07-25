import os

class RoleAssignerValidator:

    pdf_to_roles_map = {
        "~57594187-DialogFlow (WIP)-160625-170215.pdf": ["user", "admin"],
        "PRODM-Call Flows-250725-101826.pdf": ["admin"],
        "PRODM-International SMS-250725-101002.pdf": ["engineer"],
        "PRODM-Net Promoter Score (NPS)-250725-101932.pdf": ["engineer"],
        "PRODM-PM Process and Gates-250725-102254.pdf": ["engineer"],
        "PRODM-Termine 2.0 Voice-250725-101726.pdf": ["engineer"],
        "PRODM-Termine dynamisch (intelligenten Terminmanagement)-250725-101400.pdf": ["engineer"]
    }

    def __init__(self, directory: str):
        """
        Args:
            directory (str): Path to the directory containing PDF files.
        """
        self.directory = directory

    def validate_role_assignments(self):
        """
        Validates that every PDF in the directory has a role assignment in the map.

        Returns:
            tuple:
                - list of valid filenames
                - list of missing files (PDFs in dir with no role mapping)
                - list of unknown files (files in map but not present in dir)
        """
        # 1. Get list of PDF filenames in the directory
        actual_pdfs = set(f for f in os.listdir(self.directory) if f.lower().endswith(".pdf"))

        # 2. Get filenames from the role map
        mapped_pdfs = set(self.pdf_to_roles_map.keys())

        # 3. Check for mismatches
        missing_in_map = sorted(actual_pdfs - mapped_pdfs)
        unknown_in_dir = sorted(mapped_pdfs - actual_pdfs)
        valid = sorted(actual_pdfs & mapped_pdfs)

        return valid, missing_in_map, unknown_in_dir
