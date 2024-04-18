The files in this directory are files associated with the "Muncie" HEC-RAS model made available by the US Army Corps of Engineers (USACE) Hydrologic Engineering Center (HEC), and were developed for the purpose of testing rashdf code.

The original "Muncie" model can be downloaded as part of a larger archive of exmaple data via the following link:
 - https://github.com/HydrologicEngineeringCenter/hec-downloads/releases/download/1.0.31/Example_Projects_6_5.zip

The files included here are:
- Muncie.g05
- Muncie.g05.hdf
- Muncie.u02
- projection.prj

To reproduce the complete HEC-RAS model used for testing:
1. Download the "Muncie" model using the link above and open using HEC-RAS version 6.5.
2. Redefine the projection of the model within RAS Mapper using the "./projection.prj" file, based on the instruction at this link:
   - https://www.hec.usace.army.mil/confluence/rasdocs/hgt/6.5/guides/re-projecting-model-geometry
3. Associate the geometry and unsteady flow files listed above (Muncie.g05, Muncie.u02) with the model.
4. Create a HEC-RAS plan referencing the newly added geometry and unsteady flow files.