Here’s the concrete way to hook an **Ultra96-V2 (AES-ULTRA96-V2-G)** to your PC so you can **run a Vitis application** (JTAG “Launch on Hardware”):

## What you need

* **Avnet USB-to-JTAG/UART Pod** (AES-ACC-U96-JTAG). Ultra96-V2 does **not** have on-board USB-JTAG/UART; the pod plugs into headers on the board and gives you one micro-USB to the PC. ([avnet.com][1])
* A **micro-USB cable** (PC ↔ pod).
* The **12 V power supply** that came with the board.

## Cabling & switches

1. **Plug the pod into the board**

   * Pod’s **JTAG** side to **J3** (1×8 JTAG header).
   * Pod’s **UART** ribbon to **J1** (UART header).
     Be mindful of pin 1/orientation; the pod is designed to mate directly. ([avnet.com][2])
2. **USB to PC**: micro-USB from the pod to your computer. Windows should enumerate a **Xilinx JTAG** device and a **USB Serial (COMx)** for UART. ([Hackster][3])
3. **Set boot mode for JTAG**: flip **SW3** to **JTAG boot** (both switches **ON**). This tells the MPSoC to wait for JTAG, which Vitis will use. ([element14 Community][4])
4. **Power on**: plug the 12 V supply and turn the board on; the **RDY LED** should light. ([Hackster][3])
5. **Open a serial terminal** on the COM port (115200 8N1) to see your app’s prints.

> Alternative: if you want to boot from **SD card** instead of JTAG, set **SW3 to SD** (SW3-1 OFF, SW3-2 ON). That’s useful for standalone demos, but for Vitis “Run/Debug on Hardware” JTAG mode is the simplest. ([avnet.com][5])

## Vitis side (quick)

1. In Vivado, **Export Hardware** (include bitstream) → get an **.xsa**.
2. In Vitis: **New Platform** from that .xsa → **New Application (standalone)** → add your `helloworld.c`.
3. **Run As → Launch on Hardware (Single Application Debug)**. Vitis will program PL, load your ELF over JTAG, and run it; watch the UART terminal for output. (Plenty of Ultra96-V2 “Hello World” walkthroughs show exactly this flow.) ([element14 Community][6], [Hackster][3])

## Common gotchas

* **No pod?** You can’t use JTAG “Launch on Hardware” without it. You can still run apps by building a boot image (FSBL + bit + app) for **SD boot**, then power-cycling with SW3 set to SD. ([avnet.com][5])
* **VMs**: if using a VM, pass through the **“Xilinx JTAG+Serial”** USB device to the guest. ([Hackster][3])
* **Drivers**: Vivado installs the cable drivers; if Windows doesn’t recognize the JTAG/COM, (re)install the Xilinx cable drivers.

If you want, tell me which Vitis/Vivado version you’re on and I’ll give you a tiny checklist tailored to that release.

[1]: https://www.avnet.com/americas/products/avnet-boards/avnet-board-families/ultra96-v2/?utm_source=chatgpt.com "Ultra96-V2 | Avnet Boards"
[2]: https://www.avnet.com/wps/wcm/connect/onesite/b85b9556-0b2a-42b3-ad6a-8dcf3eac1ff9/Ultra96-V2-HW-User-Guide-v1_3.pdf?CACHEID=ROOTWORKSPACE.Z18_NA5A1I41L0ICD0ABNDMDDG0000-b85b9556-0b2a-42b3-ad6a-8dcf3eac1ff9-nDNP5R3&MOD=AJPERES&utm_source=chatgpt.com "Ultra96-V2 Single Board Computer Hardware User's Guide"
[3]: https://www.hackster.io/BryanF/ultra96-v2-vitis-2020-2-hello-world-from-arm-a53-2d952a?utm_source=chatgpt.com "Ultra96-V2 Vitis 2020.2 Hello World from ARM A53"
[4]: https://community.element14.com/challenges-projects/design-challenges/pathprogrammable3/b/blog/posts/path-to-programmable-iii-ultra96v2-starter-application-a53-cores---vitis-2022-2-1333723867?utm_source=chatgpt.com "Ultra96v2 Starter Application A53 cores - Vitis 2022.2"
[5]: https://www.avnet.com/wps/wcm/connect/onesite/f21462c6-4997-41a2-a95e-80122b73aea9/Ultra96-V2-GSG-v2_0.pdf?CACHEID=ROOTWORKSPACE.Z18_NA5A1I41L0ICD0ABNDMDDG0000-f21462c6-4997-41a2-a95e-80122b73aea9-nDn.uLu&MOD=AJPERES&utm_source=chatgpt.com "Ultra96-V2 Getting Started Guide"
[6]: https://community.element14.com/products/devtools/avnetboardscommunity/b/blog/posts/developing-simple-applications-in-xilinx-vitis-for-ultra96-v2?utm_source=chatgpt.com "Developing Simple Applications in Xilinx Vitis for Ultra96-V2"
