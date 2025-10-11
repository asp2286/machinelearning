// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using Microsoft.ML.Internal.Utilities;

namespace Microsoft.ML
{
    /// <summary>
    /// <see cref="IRandomSource"/> implementation backed by the SIMD-optimized Mersenne Twister.
    /// </summary>
    public sealed class MersenneTwisterRandomSource : IRandomSource, IRandomDoubleBulk
    {
        private readonly MersenneTwister _mt;

        public MersenneTwisterRandomSource(int seed)
        {
            _mt = new MersenneTwister(unchecked((uint)seed));
        }

        public int Next() => Next(int.MaxValue);

        public int Next(int maxValue)
        {
            if (maxValue < 0)
                throw new ArgumentOutOfRangeException(nameof(maxValue));

            if (maxValue == 0)
                return 0;

            return (int)NextUInt32((uint)maxValue);
        }

        public int Next(int minValue, int maxValue)
        {
            if (minValue > maxValue)
                throw new ArgumentOutOfRangeException(nameof(minValue));

            if (minValue == maxValue)
                return minValue;

            uint range = (uint)((long)maxValue - minValue);
            return (int)(NextUInt32(range) + minValue);
        }

        public long NextInt64() => NextInt64(long.MaxValue);

        public long NextInt64(long maxValue)
        {
            if (maxValue <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxValue));

            return (long)NextUInt64((ulong)maxValue);
        }

        public long NextInt64(long minValue, long maxValue)
        {
            if (minValue > maxValue)
                throw new ArgumentOutOfRangeException(nameof(minValue));

            if (minValue == maxValue)
                return minValue;

            ulong range = unchecked((ulong)maxValue - (ulong)minValue);
            ulong offset = NextUInt64(range);
            return unchecked((long)(offset + (ulong)minValue));
        }

        public double NextDouble()
        {
            Span<double> buffer = stackalloc double[1];
            _mt.NextDoubles(buffer);
            return buffer[0];
        }

        public float NextSingle()
        {
            return (float)NextDouble();
        }

        public void NextBytes(Span<byte> buffer)
        {
            if (buffer.Length == 0)
                return;

            int completeUInt32s = buffer.Length / sizeof(uint);
            if (completeUInt32s > 0)
            {
                var uintSpan = MemoryMarshal.Cast<byte, uint>(buffer[..(completeUInt32s * sizeof(uint))]);
                _mt.NextTemperedUInt32(uintSpan);
            }

            int offset = completeUInt32s * sizeof(uint);
            if (offset < buffer.Length)
            {
                uint tail = _mt.NextTemperedUInt32();
                Span<byte> tailBytes = stackalloc byte[sizeof(uint)];
                BinaryPrimitives.WriteUInt32LittleEndian(tailBytes, tail);
                tailBytes[..(buffer.Length - offset)].CopyTo(buffer[offset..]);
            }
        }

        public void NextDoubles(Span<double> destination)
        {
            _mt.NextDoubles(destination);
        }

        public void NextTemperedUInt32(Span<uint> destination)
        {
            _mt.NextTemperedUInt32(destination);
        }

        private uint NextUInt32(uint maxExclusive)
        {
            if (maxExclusive == 0)
                throw new ArgumentOutOfRangeException(nameof(maxExclusive));

            uint limit = unchecked((uint)(uint.MaxValue - (uint.MaxValue % maxExclusive)));
            while (true)
            {
                uint value = _mt.NextTemperedUInt32();
                if (value <= limit)
                    return value % maxExclusive;
            }
        }

        private ulong NextUInt64()
        {
            ulong high = _mt.NextTemperedUInt32();
            ulong low = _mt.NextTemperedUInt32();
            return (high << 32) | low;
        }

        private ulong NextUInt64(ulong maxExclusive)
        {
            if (maxExclusive == 0)
                throw new ArgumentOutOfRangeException(nameof(maxExclusive));

            ulong limit = ulong.MaxValue - (ulong.MaxValue % maxExclusive);
            while (true)
            {
                ulong value = NextUInt64();
                if (value <= limit)
                    return value % maxExclusive;
            }
        }
    }
}
