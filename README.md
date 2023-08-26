# udacity_repo
This is a repository to gain familiarity with git and Github.
Version Control with Git
https://learn.udacity.com/courses/ud123

mkdir -p udacity-git-course/new-git-project && cd $_

-> create a new, empty repository in the current directory
git init

Initializing a Repository in an Existing Directory
https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository#Initializing-a-Repository-in-an-Existing-Directory

git init docs
https://git-scm.com/docs/git-init

git init Tutorial
https://www.atlassian.com/git/tutorials/setting-up-a-repository

Git Internals - Plumbing and Porcelain
https://git-scm.com/book/en/v2/Git-Internals-Plumbing-and-Porcelain

Customizing Git - Git Hooks
https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks

the documentation for git clone
https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository#Cloning-an-Existing-Repository

git clone Tutorial
https://www.atlassian.com/git/tutorials/setting-up-a-repository


-----------------------------------------------------------------
-> Clone your repository to your local machine.
git clone url_to_github_repository

-> After making changes, moving these changes from your local back to Github.
git add name_of_file_to_move_to_github second_file_to_move_to_github
git commit -m 'added new files'
git push

----------------------------------------------------------------

-> -> Step 1: You have a local version of this repository on your laptop, and to get the latest stable version, you pull from the develop branch.

-> Switch to the develop branch
git checkout develop

-> Pull the latest changes in the develop branch
git pull

-> -> Step 2: When you start working on this demographic feature, you create a new branch called demographic, and start working on your code in this branch.

-> Create and switch to a new branch called demographic from the develop branch
git checkout -b demographic

-> Work on this new feature and commit as you go
git commit -m 'added gender recommendations'
git commit -m 'added location specific recommendations'
...


-> -> Step 3: However, in the middle of your work, you need to work on another feature. So you commit your changes on this demographic branch, and switch back to the develop branch.

-> Commit your changes before switching
git commit -m 'refactored demographic gender and location recommendations '

-> Switch to the develop branch
git checkout develop

-> -> Step 4: From this stable develop branch, you create another branch for a new feature called friend_groups.

-> Create and switch to a new branch called friend_groups from the develop branch
git checkout -b friend_groups

-> -> Step 5: After you finish your work on the friend_groups branch, you commit your changes, switch back to the development branch, merge it back to the develop branch, and push this to the remote repository’s develop branch.

-> Commit your changes before switching
git commit -m 'finalized friend_groups recommendations '

-> Switch to the develop branch
git checkout develop

-> Merge the friend_groups branch into the develop branch
git merge --no-ff friends_groups

-> Push to the remote repository
git push origin develop

-> -> Step 6: Now, you can switch back to the demographic branch to continue your progress on that feature.

-> Switch to the demographic branch
git checkout demographic

-----------------------------------------------------------------


-> shows all the branches you have available
git branch

-> reates a branch as well as moves you onto it
git checkout -b <branch_name>

-> move from one branch to another when you are not also looking to create the branch
git checkout <branch_name>

-> allows you to delete branches
git branch -d <branch_name>

-----------------------------------------------------------------

-> -> Step 1: Andrew commits his changes to the documentation branch, switches to the development branch, and pulls down the latest changes from the cloud on this development branch, including the change I merged previously for the friends group feature.

-> Commit the changes on the documentation branch
git commit -m "standardized all docstrings in process.py"]


-> Pull the latest changes on the develop branch down
git pull

-> -> Step 2: Andrew merges his documentation branch into the develop branch on his local repository, and then pushes his changes up to update the develop branch on the remote repository.

-> Merge the documentation branch into the develop branch
git merge --no-ff documentation

-> Push the changes up to the remote repository
git push origin develop

-> -> Step 3: After the team reviews your work and Andrew's work, they merge the updates from the development branch into the master branch. Then, they push the changes to the master branch on the remote repository. These changes are now in production.

-> Merge the develop branch into the master branch
git merge --no-ff develop

-> Push the changes up to the remote repository
git push --set-upstream origin master


https://nvie.com/posts/a-successful-git-branching-model/

-----------------------------------------------------------
