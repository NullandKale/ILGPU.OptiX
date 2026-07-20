// ---------------------------------------------------------------------------------------
//                                     ILGPU OptiX
//                        Copyright (c) 2020-2022 ILGPU Project
//                                    www.ilgpu.net
//
// File: HitGroupEntry.cs
//
// This file is part of ILGPU and is distributed under the University of Illinois Open
// Source License. See LICENSE.txt for details.
// ---------------------------------------------------------------------------------------

namespace ILGPU.OptiX.Pipeline
{
    /// <summary>
    /// Names for hit groups declared on a <see cref="RayTypeBuilder{TLaunchParams}"/>.
    /// </summary>
    public static class HitGroupKind
    {
        /// <summary>
        /// The kind used implicitly by <see cref="RayTypeBuilder{TLaunchParams}.HitGroup{TMaterial}(System.Action{TLaunchParams}, System.Action{TLaunchParams}?, System.Action{TLaunchParams}?)"/>
        /// and by <see cref="RayTracingPipeline{TLaunchParams}.SetHitRecords{TMaterial}(System.ReadOnlySpan{TMaterial}, int)"/> -
        /// the single-hit-group-per-ray-type case.
        /// </summary>
        public const string Default = "default";
    }

    /// <summary>
    /// One hit-group record to be packed by
    /// <see cref="RayTracingPipeline{TLaunchParams}.SetHitRecords{TMaterial}(System.ReadOnlySpan{HitGroupEntry{TMaterial}}, int)"/> -
    /// pairs the record's per-material data with the name of the
    /// <see cref="RayTypeBuilder{TLaunchParams}.HitGroup{TMaterial}(string, System.Action{TLaunchParams}, System.Action{TLaunchParams}?, System.Action{TLaunchParams}?)"/>
    /// declaration whose compiled program group should back this record. Lets one ray
    /// type mix multiple hit groups - e.g. one intersection program per custom-primitive
    /// kind, all sharing the same closest-hit/any-hit device functions - addressed
    /// purely by name, with no change to how <see cref="OptixAccelBuilder"/> assigns
    /// SBT-index offsets to geometry (that stays entirely the caller's responsibility,
    /// matching every other builder in this library).
    /// </summary>
    public readonly struct HitGroupEntry<TMaterial> where TMaterial : unmanaged
    {
        /// <summary>
        /// The hit-group name, matching a <c>kind</c> passed to
        /// <see cref="RayTypeBuilder{TLaunchParams}.HitGroup{TMaterial}(string, System.Action{TLaunchParams}, System.Action{TLaunchParams}?, System.Action{TLaunchParams}?)"/>
        /// on every ray type this pipeline declares.
        /// </summary>
        public string Kind { get; }

        /// <summary>
        /// The per-record material data.
        /// </summary>
        public TMaterial Data { get; }

        public HitGroupEntry(string kind, TMaterial data)
        {
            Kind = kind;
            Data = data;
        }
    }
}
