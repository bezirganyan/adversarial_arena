#!/usr/bin/env bash

set -o errexit

if [ $1 == "--stash" ]; then
  git stash && git pull
  (cd ../improved_wasserstein && git stash && git pull)
  (cd ../fast-wasserstein-adversarial && git stash && git pull)
else
  git pull
  (cd ../improved_wasserstein && git pull)
  (cd ../fast-wasserstein-adversarial && git pull)
fi