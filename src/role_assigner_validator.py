import os

class RoleAssignerValidator:

    heading_list = [
        "PUBLIC",
        "CONFIDENTIAL"
    ]

    def __init__(self, directory: str):
        """
        Args:
            directory (str): Path to the directory containing PDF files.
        """
        self.directory = directory

    def validate_pdfs(self):
        """
        Lists all PDF files found in the directory.

        Returns:
            list of PDF filenames found in the directory.
        """
        return sorted(f for f in os.listdir(self.directory) if f.lower().endswith(".pdf"))
