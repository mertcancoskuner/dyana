#!/bin/bash

PACKAGES=(
    "aims-chain"
    "visualdsa"
    "masterzdran-azure-tablestorge-logging"
    "inchatvx"
    "llmcode-chat"
    "aiman-client"
    "unique-linear-solver"
    "clix-bshg"
    "input-logger"
    "upfilelive"
    "yudkow-models"
    "mvent"
    "tcmath"
    "srtranslator-bariskeser"
    "js2py2"
)

for pkg in "${PACKAGES[@]}"; do
    echo "Testing package: $pkg"
    dyana trace --loader pip --package "$pkg" > "test_results/${pkg}.json" 2>&1
    echo "Completed testing package: $pkg"
    echo "----------------------------------------"
done