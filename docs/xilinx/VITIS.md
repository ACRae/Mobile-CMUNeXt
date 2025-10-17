## Vivado Design Import: XSA Import

### Step 1: Create Platform Project First
1. **Open Vitis 2025.1**
2. **Create Platform Project**:
   - Go to `File` → `New` → `Platform Project`
   - **Platform project name**: `my_platform` (or descriptive name)
   - **Choose "Create from hardware specification (XSA)"**
   - **Browse** to your `.xsa` file
   - Click `Next`

3. **Platform Configuration**:
   - **Operating System**: Select `standalone` for bare-metal or `linux` if needed
   - **Processor**: Usually auto-detected from XSA
   - **Architecture**: Should be auto-filled
   - Click `Finish`

### Step 2: Build the Platform
1. **Build Platform Project**:
   - Right-click on platform project → `Build Project`
   - Wait for platform generation to complete
   - This creates the `.xpfm` platform file

### Step 3: Create Application Project
1. **Create Application Project**:
   - `File` → `New` → `Application Project`
   - **Create a new system project** or use existing
   - **Platform**: Select `Create a new platform from hardware (XSA)`
   - Browse to your XSA file, OR better yet:
   - Select your already-created platform project

2. **Application Details**:
   - Name your application
   - Select appropriate processor
   - Choose template (Hello World)

## Common Issues and Solutions

### Issue 1: XSA File Not Recognized
- **Check XSA file**: Ensure it's properly exported from Vivado
- **Vivado Export**: In Vivado, use `File` → `Export` → `Export Hardware` with bitstream included

### Issue 2: Missing Processors in XSA
- Verify your Vivado design includes a processor (Zynq PS, MicroBlaze, etc.)
- Re-export XSA from Vivado if needed

### Issue 3: Platform Creation Fails
- Check Vitis log: `Help` → `Vitis Log`
- Common causes:
  - Corrupted XSA file
  - Missing clock or reset connections in hardware design
  - Invalid processor configuration

### Issue 4: Cannot Find Platform
If you're still having issues:
1. **Refresh Workspace**: `File` → `Refresh`
2. **Clean Workspace**: `Project` → `Clean`
3. **Check Platform Path**: Ensure `.xpfm` file was generated in platform project

## Alternative Method: Direct XSA Import in Application Project

If the above doesn't work, try:
1. **Create Application Project**
2. **Platform Selection**: Choose "Create a new platform from hardware (XSA)"
3. **Browse to XSA file** directly
4. Let Vitis create the platform automatically

## Debugging Steps

1. **Verify XSA Contents**:
   - XSA should contain hardware specification
   - Check if exported with bitstream
   - Ensure processor is properly configured in Vivado

2. **Check Vitis Version Compatibility**:
   - XSA from Vivado 2025.1 should work with Vitis 2025.1
   - Cross-version compatibility might cause issues

3. **Workspace Issues**:
   - Try creating a new workspace
   - Import XSA in fresh environment
   