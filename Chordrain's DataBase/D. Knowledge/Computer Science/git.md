# Git

Git 的核心是两大库——本地库和远程库。首先当你对一个本地目录初始化完成后，git 就会在你的硬盘的另一块地方开辟一块空间，当你想要保存对当前文件夹做出的修改时，用 add 和 commit 命令进行提交，随后这些文件会被保存到 git 在另一处地方创建的本地仓库。Git 还能连接远程仓库，例如 github、gitee 等网站的仓库，这要依靠 push 指令完成。

本文档将简单介绍使用 git 从本地上传文件到 github 库中的基本流程。

## 01 init

命令：`git init`

该命令的作用是初始化本地目录，相当于是将本地库与 git 相关联，核心是建立 `.git` 文件，不执行这一步无法进行任何 git 操作。

## 02 add

要想将本地目录中的文件上传到远程库中，必须先将本地文件添加到 git 的暂存区，指令如下：

* `git add .` 上传目录下全部文件
* `git add 文件名` 只上传指定文件

## 03 commit

Commit 命令提交更改到本地仓库，并提供相关的提交说明。

* `git commit -m "说明"`

## 04 origin

要想把本地库的文件上传到远程库，就必须声明是哪个远程库，通过设置 git 的 origin 来绑定远程库。绑定方法为：

* `git remote add origin 远程库地址`

如果要更改 origin 请使用命令：

* `git remote set-url origin 远程库地址`

查看当前 origin：

* `git remote -v`

## 05 push

通过 push 可以将已经上传到本地仓库的文件保存到远程仓库：

* `git push origin branch --force` 是否要添加 `--force` 选项要看情况，有时候会因为 github 怕你上传的文件会覆盖远程库中原有的 readme 等文件而报错，要想忽视这些报错继续上传则要加上该选项。

## 06 报错

`git branch -m master main`

## 07 clone

Clone 与上传本地仓库到远程仓库无关，它的作用是将其他人的仓库下载到我们自己的电脑上，使用方法是：

* `git clone 仓库url`