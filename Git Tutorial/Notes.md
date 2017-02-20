Introduction
============

If you are not using a version control system, you are making your life complicated. Manuscripts, code, and pretty much everything you produce as a scientist should go under a version control system. The simplest way to think to think about it is that a version control system gives you an unlimited undo history. Instead of naming your files whatever_version1.tex, ..., whatever_version678623_2.tex, a version control system allows you record each edit and compare them easily. The second advantage is that it makes collaboration very easy. The current most widespread version control system is git, which was originally developed for streamlining the workflow on the Linux kernel, and it is still used for that, and in about ten million other projects.

Contemporary version control systems, including git, are distributed. This means that you always work on the local copy of a project (called a *repository* in git). If you are working alone on a manuscript or a bit of code, that is fine. If you want to collaborate with others, you need a host for your repository. GitHub is the biggest such host for open source projects, but, for instance, Overleaf, an online collaborative editor for TeX files, also uses git under the hood.

We structure this tutorial along the following topics:

1. Using git as an annotated, unlimited undo history, with possibility for parallel branches and merges between them.

2. Using git for collaboration: hosting a repository, cloning, pushing, pulling, conflict resolution.

3. Overleaf, where git and LaTeX meet for great benefit human race.

4. GitHub, open source projects, forking, public repositories, pull requests.

5. Best practices.

The tutorial is, in principle, agnostic to the operating system that you use.


Global configuration
====================
Before you start, you must have a GitHub account. Technically you do not need a GitHub account for using git, but the way we configure, we assume the existence of an account for reasons that will become clear later. Assuming that you are Jordi Campesinyo, and your GitHub account is jordicampesinyo, do this:

```bash
git config --global user.email "jordicampesinyo@users.noreply.github.com"
git config --global user.name "Jordi Campesinyo"
```

While we are at it, log in to GitHub, and go to Settings->Emails. Check the box "Keep my email address private." We set the above email address to match the anonymous email address here. This connection ensures that your future commits will be credited to you.

Git as an unlimited undo history
================================
Create a folder called `whatever` and a file in it called `anything.txt`. In this folder, open a terminal a command prompt. Initialize your git repository and add this new file to track:

```bash
git init
git add anything.txt
```

From now on, git tracks all changes done to `anything.txt`. Add some text to this file in your favourite text editor and save it. Then make a *commit*:

```bash
git commit -am "Added initial text"
```

The message that you add is basically in annotation in the history of your edits. How frequently you make commits, that is, how frequently you add these labels, is up to you. Note that until a commit is made, your changes are not saved to the history that git maintains.

You can view the history of your edits with this command:

```bash
git log
```

If you want to see what exactly changed between edits, you can do this:

```bash
git log -p -1
```

where `-1` refers to the length of history you wish to review.

**Exercise 1**. Edit the text a bit more, make one more commit, and review the changes in the last two commits.

It often happens that you want to explore a parallel idea that may or may not make it to the final version of whatever you are working on. It could be a section in a manuscript or some tricky bit of coding. To help working with these explorations, git allows you to have parallel histories by creating *branches*. Let us create a new branch and switch to it:

```bash
git branch Curses
git checkout Curses
```

Now every commit you make will go to this branch alone. Add some curses to `anything.txt`, and commit:

```bash
git commit -am "Some fine curses added"
```

Now switch back to the main branch:

```bash
git checkout master
```

If you look at the text file or the commit history, there is no sign of curses. You can continue editing undisturbed if you want to. Then, at any point in time, you can go back to the curses branch:

```bash
git checkout Curses
```

If you finally made up your mind that the curses should be a part of your text, you can *merge* this branch with master:

```bash
git merge master Curses
```

Now if you go back to master, you will see the merge:

```bash
git checkout master
git log
```


Collaboration: An example through Overleaf
==========================================
Go to [Overleaf](https://overleaf.com/) and hit "Create a New Paper." No registration is necessary. You will be directed to a new LaTeX manuscript, and the URL will be something similar to this one:

[https://www.overleaf.com/8253629qgvxqbvqvyzc#/29202944/](https://www.overleaf.com/8253629qgvxqbvqvyzc#/29202944/)

You can start sending around this link to co-authors to work on the manuscript collaboratively. But this is not want we want. This new manuscript is actually a git repository. You can *clone* it by removing everything from the link above starting with `#`, and changing `www` to `git`:

```bash
git clone https://git.overleaf.com/8253629qgvxqbvqvyzc
```

Cloning means you make a local copy of the repository. You can rename the cloned repository to anything. In this case, it has the name `8253629qgvxqbvqvyzc`, but you can call it "The best Nature-candidate manuscript ever". It does not matter. Change into the directory. From the directory, you can do everything you did in the previous section: add text, commit, and branch.

How do you make your changes available to others? You *push* your commits:

```bash
git push
```

How do you receive the changes others made? You *pull* their commits (that they already pushed):

```bash
git pull
```

If someone pushed before you, git will ask you to pull first anyway.

Pull actually has two phases: *fetching* and merging. Fetch means bringing the changes to your local copy of the repository, and merging is the same as when you were merging branches. A pull works fine if there is no conflict. If there is, you have to resolve it. The strength of git is the efficiency of automatic conflict resolution and the help it gives you for manual resolutions.

*Exercise 2*. Work in pairs on the same Overleaf repository. Create a clone each, and create a conflicting edit by making changes on the same line. Push your local changes and try pulling. Resolve the conflict.

You can manually separate pulling into its two phases, which is often necessary when many people work on the same project:

```bash
git fetch
git merge
```

Trick: technically, Overleaf is for LaTeX projects. The repositories are not public: you must know the URL to access it. This makes them semi-private. You can add arbitrary files to repository (e.g. Python scripts) and use it as it would any other git repository. It is a cheap way to work on secretive projects with others.

GitHub
======
This is the real deal. GitHub is the best place to host open source projects and collaborate on them. It made contributions easy, introducing the idea of "social coding." The downside: all repositories are public, unless you pay. This is why the trick with Overleaf is handy for giving you a free private repository.

We need to introduce three more concepts. The first one is *forking*: it means that you copy somebody else's repository. It is like the branches before, except that branches are within a local copy of a repository, whereas forking duplicates repositories. Then *upstream* is the original repository that you forked. If you make changes to your fork and want to merge them back to upstream, you send a *pull request* (PR). Forking and sending PRs happen on GitHub, not on the git command line.

If you have not done it yet, go to [https://github.com/peterwittek/qml-rg](https://github.com/peterwittek/qml-rg) and hit fork. Creating a fork is exactly that complicated. Now clone it to have a local copy of **your fork**, and not the upstream:

```bash
git clone https://github.com/jordicampesinyo/qml-rg
```

So far so good. Now you can make edits, and so and so forth. If


Best practices
==============

- In LaTeX files, keep every sentence a new line. This ensures easy conflict resolution.

- If you like or use an open source project on GitHub, give it a star. It means a lot to the developers, both in terms of self-glorification, and also in terms of job prospects. The GitHub profile is an integral part of a computer science CV, and it is akin to your academic citation index.

