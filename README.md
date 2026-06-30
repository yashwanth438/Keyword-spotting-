# Keyword-spotting-
This project implements an offline Keyword Spotting (KWS) system using the nRF5340 DK and a PDM microphone (MP34DT01-M). The firmware continuously captures audio, extracts MFCC (Mel-Frequency Cepstral Coefficients) features using CMSIS-DSP, and feeding the mfcc to the ML model running in PC.

The project also supports Firmware Over-The-Air (FOTA) updates using MCUboot, MCUmgr, and Bluetooth Low Energy (BLE). The secondary firmware image is stored in an external SPI NOR flash (MX25R6435F).

| Component         | Description                        |
| ----------------- | ---------------------------------- |
| Development Board | Nordic nRF5340 DK                  |
| Microphone        | MP34DT01-M PDM Microphone          |
| External Flash    | MX25R6435F (64 Mbit SPI NOR Flash) |
| Debugger          | On-board J-Link                    |
| Power             | USB                                |


pdm_mic_5340/
│
├── boards/
│   └── nrf5340dk_nrf5340_cpuapp.overlay
│
├── child_image/
│   └── mcuboot.conf
│
├── ML/
│   ├── kws_model.h5
│   └── kws_model_int8.tflite
│
├── speech_commands/
│
├── src/
│   ├── KWS_main.c
│   ├── model_data.cc
│   ├── feature_provider.cpp
│   ├── recognize_commands.cpp
│   └── ...
│
├── prj.conf
├── pm_static.yml
├── CMakeLists.txt
└── README.md
