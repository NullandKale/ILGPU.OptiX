using System;
using System.IO;
using System.Text;
using AsmResolver.PE.File;

namespace GpuExportPatcher
{
    /// <summary>
    /// Post-build tool: patches a built .NET apphost .exe to export
    /// <c>NvOptimusEnablement</c> and <c>AmdPowerXpressRequestHighPerformance</c> as
    /// DWORD data symbols, so NVIDIA/AMD drivers route the process to the discrete
    /// GPU on hybrid-graphics laptops automatically - the same hint C++ apps embed
    /// via <c>__declspec(dllexport)</c>, which the C# compiler has no equivalent for.
    ///
    /// Modern .NET (5+) launches a generic native "apphost" stub that loads the
    /// managed assembly as a DLL, rather than compiling your code into the .exe
    /// directly (as .NET Framework did) - so there is no source-level way to add
    /// this export. This tool adds it after the fact by appending a new PE section
    /// containing a hand-built IMAGE_EXPORT_DIRECTORY (a small, stable, well-known
    /// Win32 structure) directly, rather than depending on AsmResolver's high-level
    /// export-building APIs, to keep the on-disk layout fully deterministic and
    /// easy to verify by inspection.
    ///
    /// Usage: GpuExportPatcher &lt;path-to-exe&gt;
    /// </summary>
    public static class Program
    {
        private const string AmdSymbolName = "AmdPowerXpressRequestHighPerformance";
        private const string NvSymbolName = "NvOptimusEnablement";

        public static int Main(string[] args)
        {
            if (args.Length != 1)
            {
                Console.Error.WriteLine("Usage: GpuExportPatcher <path-to-exe>");
                return 1;
            }

            string exePath = args[0];
            if (!File.Exists(exePath))
            {
                Console.Error.WriteLine($"GpuExportPatcher: file not found: {exePath}");
                return 1;
            }

            var peFile = PEFile.FromFile(exePath);

            // Pass 1: reserve a section of the final size so UpdateHeaders() assigns
            // it a real RVA - the export directory's internal fields are RVAs of
            // other data in this same section, so we need to know our own base RVA
            // before we can fill them in.
            int contentSize = ComputeContentSize();
            var section = new PESection(
                new AsmResolver.Utf8String(".gpuopt"),
                SectionFlags.MemoryRead | SectionFlags.ContentInitializedData,
                new AsmResolver.DataSegment(new byte[contentSize]));
            peFile.Sections.Add(section);
            peFile.UpdateHeaders();

            // Pass 2: now that section.Rva is known, build the real bytes and set
            // the Export data directory to point at the IMAGE_EXPORT_DIRECTORY
            // struct within it.
            uint sectionRva = section.Rva;
            var (content, exportDirOffset, exportDirSize) = BuildContent(sectionRva);
            if (content.Length != contentSize)
                throw new InvalidOperationException(
                    $"Internal error: computed size {contentSize} but built {content.Length} bytes.");
            section.Contents = new AsmResolver.DataSegment(content);

            peFile.OptionalHeader.SetDataDirectory(
                DataDirectoryIndex.ExportDirectory,
                new DataDirectory(sectionRva + exportDirOffset, exportDirSize));

            peFile.UpdateHeaders();
            peFile.Write(exePath);

            Console.WriteLine(
                $"GpuExportPatcher: added NvOptimusEnablement/AmdPowerXpressRequestHighPerformance exports to {Path.GetFileName(exePath)}");
            return 0;
        }

        // ---- Byte layout (all offsets relative to the start of the section) ----
        //
        //  0   AmdValue            DWORD = 1
        //  4   NvValue             DWORD = 1
        //  8   dllName             "gpuopt\0"                              (8 bytes)
        //  16  name0 (sorted)      "AmdPowerXpressRequestHighPerformance\0" (38 bytes)
        //  54  name1 (sorted)      "NvOptimusEnablement\0"                 (20 bytes)
        //  74  padding to 4-byte alignment                                 (2 bytes)
        //  76  AddressOfFunctions  DWORD[2] = { AmdValue RVA, NvValue RVA }
        //  84  AddressOfNames      DWORD[2] = { name0 RVA, name1 RVA }     (sorted)
        //  92  AddressOfNameOrdinals WORD[2] = { 0, 1 }
        //  96  IMAGE_EXPORT_DIRECTORY (40 bytes)
        //  136 (end)
        //
        // Ordinal 0 = Amd (functions[0]), ordinal 1 = Nv (functions[1]) - the name
        // ordinal array maps each (sorted) name back to its slot in the functions
        // array, which is what the loader's GetProcAddress binary search walks.

        private const string DllName = "gpuopt";

        private static int ComputeContentSize() => BuildContent(0).Content.Length;

        private static (byte[] Content, uint ExportDirOffset, uint ExportDirSize) BuildContent(uint sectionRva)
        {
            var amdNameBytes = Encoding.ASCII.GetBytes(AmdSymbolName + "\0");
            var nvNameBytes = Encoding.ASCII.GetBytes(NvSymbolName + "\0");
            var dllNameBytes = Encoding.ASCII.GetBytes(DllName + "\0");

            const int amdValueOffset = 0;
            const int nvValueOffset = 4;
            int dllNameOffset = 8;
            int name0Offset = dllNameOffset + dllNameBytes.Length;
            int name1Offset = name0Offset + amdNameBytes.Length;
            int afterNames = name1Offset + nvNameBytes.Length;
            int addressOfFunctionsOffset = Align4(afterNames);
            int addressOfNamesOffset = addressOfFunctionsOffset + 8;
            int addressOfNameOrdinalsOffset = addressOfNamesOffset + 8;
            int exportDirOffset = addressOfNameOrdinalsOffset + 4;
            int totalSize = exportDirOffset + 40;

            var buffer = new byte[totalSize];

            WriteUInt32(buffer, amdValueOffset, 1);
            WriteUInt32(buffer, nvValueOffset, 1);
            Array.Copy(dllNameBytes, 0, buffer, dllNameOffset, dllNameBytes.Length);
            Array.Copy(amdNameBytes, 0, buffer, name0Offset, amdNameBytes.Length);
            Array.Copy(nvNameBytes, 0, buffer, name1Offset, nvNameBytes.Length);

            // AddressOfFunctions: ordinal 0 -> Amd value, ordinal 1 -> Nv value.
            WriteUInt32(buffer, addressOfFunctionsOffset + 0, sectionRva + amdValueOffset);
            WriteUInt32(buffer, addressOfFunctionsOffset + 4, sectionRva + nvValueOffset);

            // AddressOfNames: must be sorted ascending by name for GetProcAddress's
            // binary search ("Amd..." < "Nv..." lexicographically).
            WriteUInt32(buffer, addressOfNamesOffset + 0, sectionRva + (uint)name0Offset);
            WriteUInt32(buffer, addressOfNamesOffset + 4, sectionRva + (uint)name1Offset);

            // AddressOfNameOrdinals: name0 ("Amd...") -> ordinal 0, name1 ("Nv...") -> ordinal 1.
            WriteUInt16(buffer, addressOfNameOrdinalsOffset + 0, 0);
            WriteUInt16(buffer, addressOfNameOrdinalsOffset + 2, 1);

            // IMAGE_EXPORT_DIRECTORY
            WriteUInt32(buffer, exportDirOffset + 0, 0);   // Characteristics
            WriteUInt32(buffer, exportDirOffset + 4, 0);   // TimeDateStamp
            WriteUInt16(buffer, exportDirOffset + 8, 0);   // MajorVersion
            WriteUInt16(buffer, exportDirOffset + 10, 0);  // MinorVersion
            WriteUInt32(buffer, exportDirOffset + 12, sectionRva + (uint)dllNameOffset); // Name
            WriteUInt32(buffer, exportDirOffset + 16, 1);  // Base (ordinal base)
            WriteUInt32(buffer, exportDirOffset + 20, 2);  // NumberOfFunctions
            WriteUInt32(buffer, exportDirOffset + 24, 2);  // NumberOfNames
            WriteUInt32(buffer, exportDirOffset + 28, sectionRva + (uint)addressOfFunctionsOffset);    // AddressOfFunctions
            WriteUInt32(buffer, exportDirOffset + 32, sectionRva + (uint)addressOfNamesOffset);        // AddressOfNames
            WriteUInt32(buffer, exportDirOffset + 36, sectionRva + (uint)addressOfNameOrdinalsOffset); // AddressOfNameOrdinals

            return (buffer, (uint)exportDirOffset, 40u);
        }

        private static int Align4(int value) => (value + 3) & ~3;

        private static void WriteUInt32(byte[] buffer, int offset, uint value)
        {
            buffer[offset + 0] = (byte)(value & 0xFF);
            buffer[offset + 1] = (byte)((value >> 8) & 0xFF);
            buffer[offset + 2] = (byte)((value >> 16) & 0xFF);
            buffer[offset + 3] = (byte)((value >> 24) & 0xFF);
        }

        private static void WriteUInt16(byte[] buffer, int offset, ushort value)
        {
            buffer[offset + 0] = (byte)(value & 0xFF);
            buffer[offset + 1] = (byte)((value >> 8) & 0xFF);
        }
    }
}
