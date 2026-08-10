@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM ===========================================================================
REM   deploy.bat  -  commit once, publish to both GitHub and Hugging Face.
REM
REM   WHY TWO DIFFERENT PUSHES
REM   ------------------------
REM   GitHub gets the real branch, history and all.
REM
REM   Hugging Face gets a single-commit orphan branch instead. main's history
REM   contains 5,344 training images committed long ago; .gitignore does not
REM   apply retroactively to tracked files, so pushing main to a Space gets
REM   rejected by its pre-receive hook and would blow the 1GB storage limit.
REM   A Space needs the files it runs, not development history - so each deploy
REM   rebuilds a throwaway branch from the working tree and force-pushes that.
REM
REM   THIS SCRIPT MUST BE COMMITTED ON main
REM   -------------------------------------
REM   cmd.exe reads a .bat line by line off disk while it runs. The first
REM   version of this script was untracked on main but got committed onto the
REM   orphan branch, so `git checkout main` at the end deleted it mid-execution
REM   and cmd died with "The batch file cannot be found". Step 1 commits it, so
REM   both branches hold an identical copy and checkout never touches the file.
REM
REM   Replaces git-push.bat, which only knew about GitHub.
REM ===========================================================================

set "GH_REMOTE=origin"
set "HF_REMOTE=space"
set "HF_URL=https://huggingface.co/spaces/undebuggedbit/Rice-Leaf-Disease-Detector"
set "MAIN=main"
set "DEPLOY=hf-deploy"

REM Guard: the deploy branch should be the app only. Anything far above this
REM means something started being tracked that should not be - the push would
REM be rejected anyway, so stop here and say why.
set "MAX_FILES=200"

set "GH_OK=0"
set "HF_OK=0"

echo.
echo ============================================================
echo   Deploy - GitHub + Hugging Face
echo   Folder : %CD%
echo ============================================================
echo.

git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Not a git repository.
    goto :fail
)

REM A stale lock blocks every checkout. It is left behind by a crashed git or
REM by an editor's background git process, and deleting it is safe when no git
REM command is actually running.
if exist ".git\index.lock" (
    echo [setup] Removing stale .git\index.lock ...
    del /f /q ".git\index.lock" >nul 2>&1
)

git remote get-url %HF_REMOTE% >nul 2>&1
if errorlevel 1 (
    echo [setup] Adding '%HF_REMOTE%' remote...
    git remote add %HF_REMOTE% "%HF_URL%"
)

REM Single quotes, not doubled. `''yyyy-MM-dd''` is not valid PowerShell; it
REM returned nothing, `git commit -m ""` aborted, and the script reported
REM "nothing new to commit" while quietly committing nothing at all.
for /f "delims=" %%i in ('powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm'"') do set "STAMP=%%i"
if not defined STAMP set "STAMP=deploy"

REM --- 1. commit on main -----------------------------------------------------
echo [1/5] Committing on %MAIN% ^(%STAMP%^)...
git checkout %MAIN% >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Could not switch to %MAIN%. Commit or stash your changes first.
    goto :fail
)

REM Training images were tracked before .gitignore covered them. Untrack them
REM if they are still in the index; a no-op once already done.
git rm -r --cached train validation train_stage2 validation_stage2 -q >nul 2>&1

git add -A
git commit -m "%STAMP%" >nul 2>&1
if errorlevel 1 (echo       nothing new to commit) else (echo       committed)

REM --- 2. push to GitHub -----------------------------------------------------
echo [2/5] Pushing to GitHub...
git push %GH_REMOTE% %MAIN%
if errorlevel 1 (
    echo.
    echo       [WARNING] GitHub push failed. Continuing to Hugging Face anyway.
    echo       If it mentions LFS quota: https://github.com/settings/billing
    echo.
) else (
    set "GH_OK=1"
)

REM --- 3. build a clean single-commit branch ---------------------------------
echo [3/5] Building deploy branch...
git branch -D %DEPLOY% >nul 2>&1
git checkout --orphan %DEPLOY% >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Could not create the %DEPLOY% branch.
    goto :restore
)
git add -A
git commit -m "Deploy %STAMP%" >nul 2>&1

for /f %%c in ('git ls-files ^| find /c /v ""') do set "NFILES=%%c"
echo       %NFILES% files staged for the Space

if %NFILES% GTR %MAX_FILES% (
    echo.
    echo       [ABORTED] %NFILES% files is far more than a deployment needs.
    echo       Something large is being tracked - check 'git ls-files' and add
    echo       it to .gitignore before retrying.
    goto :restore
)

REM --- 4. push to Hugging Face -----------------------------------------------
echo [4/5] Pushing to Hugging Face ^(large model files take a while^)...
git push --force %HF_REMOTE% %DEPLOY%:%MAIN%
if errorlevel 1 (
    echo.
    echo       [WARNING] Hugging Face push failed. Common causes:
    echo         - Password must be a WRITE token, not your account password.
    echo           https://huggingface.co/settings/tokens
    echo         - Storage limit. Delete old LFS files in Space Settings.
    echo.
) else (
    set "HF_OK=1"
)

:restore
REM Summary is printed BEFORE returning to main. If a future edit ever leaves
REM this script differing between the two branches, the checkout below would
REM rewrite it and cmd would lose its place - so nothing important comes after.
echo.
echo ============================================================
if "%GH_OK%"=="1" (echo   GitHub          : pushed) else (echo   GitHub          : FAILED)
if "%HF_OK%"=="1" (echo   Hugging Face    : pushed) else (echo   Hugging Face    : FAILED)
echo.
echo   Space build : %HF_URL%
echo ============================================================
echo.

echo [5/5] Returning to %MAIN%...
git checkout %MAIN% >nul 2>&1
git branch -D %DEPLOY% >nul 2>&1
echo       done.
echo.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
