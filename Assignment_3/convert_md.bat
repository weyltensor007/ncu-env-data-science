@echo off

pandoc assignment_3.md -o assignment_3.pdf ^
--pdf-engine=xelatex ^
-V mainfont="Microsoft JhengHei" ^
--resource-path=.:imgs ^
-V colorlinks=true ^
-V linkcolor=blue ^
-V urlcolor=cyan ^
-V citecolor=magenta ^
-V header-includes="\usepackage{float}" && start assignment_3.pdf