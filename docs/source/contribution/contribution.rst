Information for Developers and Contributors
===========================================

OpenPIV-Python-cxx is a continuation of the OpenPIV project to create free, opensource, and fast particle image velocimetry (PIV) software. The core functions are written in c++ to help alleviate memory constraints on consumer grade laptops and desktops while remaining fast. However, this package needs help from developers to improve further.

Development Workflow
---------------------
This is not a comprehensive guide for git developement and it only serves as a template to help developers get started.


Without Write Access
--------------------

1. If not done, install Git (platform dependend) and configure it on the command line::

    git config --global user.name "first name surname"
    git config --global user.email "e-mail address"

2. Create a Github account, navigate to the `OpenPIV-cxx Github page <https://github.com/ErichZimmer/openpiv-python-cxx>`_ and press the fork button (top right of the page). Github will create a personal online fork of the repository for you.

3. Clone your fork, to get a local copy::

    git clone https://github.com/ErichZimmer/openpiv-python-cxx.git

4. Your fork is independent from the original (upstream) repository. To be able to sync changes in the upstream repository with your fork later, specify the upstream repository::

    cd openpiv_t-python-cxx
    git remote add upstream https://github.com/ErichZimmer/openpiv-python-cxx.git
    git remote -v

5. Change the code locally, commit the changes::

    git add . 
    git commit -m 'A meaningful comment on the changes.'

6. See, if there are updates in the upstream repository and save them in your local branch upstream/master:::

    git fetch upstream

7. Merge possible upstream changes into your local master branch::

    git merge upstream/master

8. If there are merge conflicts, use ``git status`` and ``git diff`` for displaying them. Git marks conflicts in your files, `as described in the Github documentation on solving merge conflicts <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/resolving-a-merge-conflict-using-the-command-line>`_. After resolving merge conflicts, upload everything::

    git add .
    git commit -m 'A meaningful comment.'
    git push

9. Propose your changes to the upstream developer by creating a pull-request, as described `in the Github documentation for creating a pull-request from a fork <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork>`_. (Basically just pressing the »New pull request« button.)


With Write Access
-----------------

1. If not done, install Git and configure it::

    git config --global user.name "first name surname"
    git config --global user.email "e-mail address"

2. Clone the git repository::

    git clone https://github.com/ErichZimmer/openpiv-python-cxx.git

3. Create a new branch and switch over to it::

    cd openpiv-python-cxx
    git branch meaningful-branch-name
    git checkout meaningful-branch-name
    git status

4. Change the code locally and commit changes::

    git add .
    git commit -m 'A meaningful comment on the changes.'

5. Push branch, so everyone can see it::

    git push --set-upstream origin meaningful-branch-name

6. Create a pull request. This is not a Git, but a Github feature, so you must use the Github user-interface, as described in the `Github documentaton on creating a pull request from a branch <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request#creating-the-pull-request>`_.

7. After discussing the changes and possibly additional commits, the feature-branch can be merged into the main branch::

    git checkout master
    git merge meaningful-branch-name

8. Eventually, solve merge conflicts. Use ``git status`` and ``git diff`` for displaying conflicts. Git marks conflicts in your files, `as described in the Github documentation on solving merge conflicts <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/resolving-a-merge-conflict-using-the-command-line>`_.

9. Finally, the feature-branch can safely be removed::

    git branch -d meaningful-branch-name

10. Go to the Github user-interface and also delete the now obsolete online copy of the feature-branch.