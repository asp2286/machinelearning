// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Buffers;

namespace Microsoft.ML
{
    /// <summary>
    /// Abstraction over random number generation used by <see cref="MLContext"/>.
    /// </summary>
    public interface IRandomSource
    {
        int Next();
        int Next(int maxValue);
        int Next(int minValue, int maxValue);
        long NextInt64();
        long NextInt64(long maxValue);
        long NextInt64(long minValue, long maxValue);
        double NextDouble();
        float NextSingle();
        void NextBytes(Span<byte> buffer);
    }

    /// <summary>
    /// Optional interface for sources that can efficiently populate buffers with double
    /// or 32-bit tempered values.
    /// </summary>
    public interface IRandomDoubleBulk
    {
        /// <summary>
        /// Fills <paramref name="destination"/> with independent <c>U[0, 1)</c> samples.
        /// </summary>
        void NextDoubles(Span<double> destination);

        /// <summary>
        /// Fills <paramref name="destination"/> with tempered 32-bit unsigned integers.
        /// </summary>
        void NextTemperedUInt32(Span<uint> destination);
    }

    internal static class RandomSourceHelpers
    {
        public static long NextInt64(Random rand)
        {
#if NET6_0_OR_GREATER
            return rand.NextInt64();
#else
            while (true)
            {
                ulong result = NextUInt64(rand) >> 1;
                if (result != long.MaxValue)
                    return (long)result;
            }
#endif
        }

        public static long NextInt64(Random rand, long maxValue)
        {
#if NET6_0_OR_GREATER
            return rand.NextInt64(maxValue);
#else
            if (maxValue <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxValue));

            return (long)NextUInt64(rand, (ulong)maxValue);
#endif
        }

        public static long NextInt64(Random rand, long minValue, long maxValue)
        {
#if NET6_0_OR_GREATER
            return rand.NextInt64(minValue, maxValue);
#else
            if (minValue > maxValue)
                throw new ArgumentOutOfRangeException(nameof(minValue));

            if (minValue == maxValue)
                return minValue;

            ulong range = unchecked((ulong)(maxValue - minValue));
            ulong sample = NextUInt64(rand, range);
            return (long)sample + minValue;
#endif
        }

        public static void NextBytes(Random rand, Span<byte> buffer)
        {
#if NET6_0_OR_GREATER
            rand.NextBytes(buffer);
#else
            if (buffer.Length == 0)
                return;

            byte[] rented = ArrayPool<byte>.Shared.Rent(buffer.Length);
            try
            {
                rand.NextBytes(rented);
                new ReadOnlySpan<byte>(rented, 0, buffer.Length).CopyTo(buffer);
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
#endif
        }

#if !NET6_0_OR_GREATER
        private static ulong NextUInt64(Random rand)
        {
            Span<byte> buffer = stackalloc byte[8];
            FillBuffer(rand, buffer);
            return ReadUInt64(buffer);
        }

        private static ulong NextUInt64(Random rand, ulong maxExclusive)
        {
            if (maxExclusive == 0)
                throw new ArgumentOutOfRangeException(nameof(maxExclusive));

            ulong remainder = ulong.MaxValue % maxExclusive;
            while (true)
            {
                ulong value = NextUInt64(rand);
                if (value <= ulong.MaxValue - remainder)
                    return value % maxExclusive;
            }
        }

        private static void FillBuffer(Random rand, Span<byte> destination)
        {
            byte[] rented = ArrayPool<byte>.Shared.Rent(destination.Length);
            try
            {
                rand.NextBytes(rented);
                new ReadOnlySpan<byte>(rented, 0, destination.Length).CopyTo(destination);
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
        }

        private static ulong ReadUInt64(ReadOnlySpan<byte> buffer)
        {
            return buffer[0]
                | ((ulong)buffer[1] << 8)
                | ((ulong)buffer[2] << 16)
                | ((ulong)buffer[3] << 24)
                | ((ulong)buffer[4] << 32)
                | ((ulong)buffer[5] << 40)
                | ((ulong)buffer[6] << 48)
                | ((ulong)buffer[7] << 56);
        }
#endif
    }

    internal sealed class RandomSourceAdapter : IRandomSource
    {
        private readonly Random _rand;

        public RandomSourceAdapter(Random rand)
        {
            _rand = rand ?? throw new ArgumentNullException(nameof(rand));
        }

        public int Next() => _rand.Next();

        public int Next(int maxValue) => _rand.Next(maxValue);

        public int Next(int minValue, int maxValue) => _rand.Next(minValue, maxValue);

        public long NextInt64() => RandomSourceHelpers.NextInt64(_rand);

        public long NextInt64(long maxValue) => RandomSourceHelpers.NextInt64(_rand, maxValue);

        public long NextInt64(long minValue, long maxValue) => RandomSourceHelpers.NextInt64(_rand, minValue, maxValue);

        public double NextDouble() => _rand.NextDouble();

        public float NextSingle() => _rand.NextSingle();

        public void NextBytes(Span<byte> buffer) => RandomSourceHelpers.NextBytes(_rand, buffer);
    }
}
