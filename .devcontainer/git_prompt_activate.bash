# This file, when sourced in ~/.bashrc, puts formatted __git_ps1 in $PS1.
# Based on: https://github.com/EliahKagan/git_prompt_activate

# shellcheck shell=bash
# shellcheck disable=SC1091  # Not checking /usr/lib/git-core/git-sh-prompt.
# shellcheck disable=SC2016  # PS1 will contain $ syntax for later expansion.

if test "$(git config devcontainers-theme.hide-status)" = 1 ||
   test "$(git config codespaces-theme.hide-status)" = 1
then
    . /usr/lib/git-core/git-sh-prompt

    git_prompt_part='$(__git_ps1 "(%s)")'

    case "$TERM" in
    xterm-color|*-256color) # See color_prompt in .bashrc.
        git_prompt_part='\[\033[36m\]'"$git_prompt_part"'\[\033[0m\]' ;;
    esac

    case "$PS1" in
    *'\$ ')
        PS1="${PS1/%\\$ /$git_prompt_part\\$ }" ;;
    *)
        PS1="${PS1/%$ /$git_prompt_part\\$ }" ;;
    esac

    unset git_prompt_part

    GIT_PS1_SHOWDIRTYSTATE=1
    GIT_PS1_SHOWSTASHSTATE=1
    GIT_PS1_SHOWUNTRACKEDFILES=1
    GIT_PS1_SHOWUPSTREAM=1
fi
