// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML
{
    internal sealed class RandomSourceDelegatingRandom : Random
    {
        private readonly IRandomSource _source;

        public RandomSourceDelegatingRandom(IRandomSource source)
            : base(0)
        {
            _source = source ?? throw new ArgumentNullException(nameof(source));
        }

        protected override double Sample() => _source.NextDouble();

        public override int Next() => _source.Next();

        public override int Next(int maxValue) => _source.Next(maxValue);

        public override int Next(int minValue, int maxValue) => _source.Next(minValue, maxValue);

        public override double NextDouble() => _source.NextDouble();

        public override void NextBytes(byte[] buffer)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));

            _source.NextBytes(new Span<byte>(buffer));
        }

#if NET6_0_OR_GREATER
        public override long NextInt64() => _source.NextInt64();

        public override long NextInt64(long maxValue) => _source.NextInt64(maxValue);

        public override long NextInt64(long minValue, long maxValue) => _source.NextInt64(minValue, maxValue);

        public override void NextBytes(Span<byte> buffer) => _source.NextBytes(buffer);

        public override float NextSingle() => _source.NextSingle();
#endif
    }
}
