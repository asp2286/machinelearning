// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML;
using Microsoft.ML.RunTests;
using Xunit;
using Xunit.Abstractions;

namespace Microsoft.ML.Core.Tests.UnitTests
{
    public sealed class MersenneTwisterTests : BaseTestBaseline
    {
        public MersenneTwisterTests(ITestOutputHelper output)
            : base(output)
        {
        }

        [Fact]
        [TestCategory("Utilities")]
        public void MixedApiCallsConsumeTemperedSequenceWithoutGaps()
        {
            const uint seed = 5489u;

            var baseline = new uint[32];
            var baselineTwister = new MersenneTwister(seed);
            baselineTwister.NextTemperedUInt32(baseline);

            var twister = new MersenneTwister(seed);
            int index = 0;

            Assert.Equal(baseline[index++], twister.NextTemperedUInt32());

            double firstDouble = twister.NextDouble();
            Assert.Equal(ToDoubleFromTempered(baseline[index], baseline[index + 1]), firstDouble);
            index += 2;

            var buffer = new uint[5];
            twister.NextTemperedUInt32(buffer);
            Assert.Equal(Slice(baseline, index, buffer.Length), buffer);
            index += buffer.Length;

            Assert.Equal(baseline[index++], twister.NextTemperedUInt32());

            double secondDouble = twister.NextDouble();
            Assert.Equal(ToDoubleFromTempered(baseline[index], baseline[index + 1]), secondDouble);
            index += 2;

            var secondBuffer = new uint[3];
            twister.NextTemperedUInt32(secondBuffer);
            Assert.Equal(Slice(baseline, index, secondBuffer.Length), secondBuffer);
            index += secondBuffer.Length;

            double thirdDouble = twister.NextDouble();
            Assert.Equal(ToDoubleFromTempered(baseline[index], baseline[index + 1]), thirdDouble);
            index += 2;

            var thirdBuffer = new uint[4];
            twister.NextTemperedUInt32(thirdBuffer);
            Assert.Equal(Slice(baseline, index, thirdBuffer.Length), thirdBuffer);
            index += thirdBuffer.Length;

            Assert.Equal(baseline[index++], twister.NextTemperedUInt32());

            Assert.True(index <= baseline.Length);
        }

        private static uint[] Slice(uint[] source, int start, int length)
        {
            var result = new uint[length];
            Array.Copy(source, start, result, 0, length);
            return result;
        }

        private static double ToDoubleFromTempered(uint first, uint second)
        {
            ulong a = (ulong)(first >> 5);
            ulong b = (ulong)(second >> 6);
            ulong mantissa = (a << 26) | b;
            long bits = unchecked((long)((1023UL << 52) | mantissa));
            return BitConverter.Int64BitsToDouble(bits) - 1.0;
        }
    }
}
