#!/usr/bin/env bash

branch=`git symbolic-ref --short HEAD`

git add *
date_string=`date`
echo $date_string
msg='git auto commit at '
git commit -m"$msg"