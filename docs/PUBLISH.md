# Personal GitHub publication handoff

This research tree has a local main branch and commits authored as:
Mr-Akhil12 <180930390+Mr-Akhil12@users.noreply.github.com>
This avoids putting a personal email address in public Git history.

Do not publish until you review the README, license and release checklist.
Prefer a PRIVATE GitHub repository first; make public only when ready.

On github.com as user Mr-Akhil12, create an empty repository named
grassmann-edge-research. Avoid adding an online README or license because
the local repository already contains these files.

Then from the project root in Windows PowerShell:

  & 'C:\Program Files\Git\cmd\git.exe' remote add origin https://github.com/Mr-Akhil12/grassmann-edge-research.git
  & 'C:\Program Files\Git\cmd\git.exe' push -u origin main

The repository URL above is a planned destination, NOT confirmed to exist yet.
GitHub may request browser authentication at push time. Never paste a token
into code or terminal history.

Before public visibility, complete docs/RELEASE_CHECKLIST.md.
Do not publish the sibling _local_setup_notes or _source_references directories.
Do not include venvs, pip caches, personally identifiable hardware snapshots,
or proprietary datasets.

The project deliberately keeps all training/model-quality claims unproven
until matched experimental protocols have passed.
