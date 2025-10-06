using System;
using System.Linq;
using System.Reflection;
using Microsoft.ML;

namespace Samples.Dynamic
{
    public static class MersenneTwister
    {
        public static void Example()
        {
            const uint seed = 5489u;

            // The MersenneTwister type is internal to Microsoft.ML.Core, so the
            // sample uses a thin reflection-based wrapper that exposes the
            // members we want to demonstrate.
            var generator = MersenneTwisterWrapper.Create(seed);

            var doubles = new double[5];
            for (var i = 0; i < doubles.Length; i++)
                doubles[i] = generator.NextDouble();

            Console.WriteLine("First five doubles from the generator:");
            foreach (var value in doubles)
                Console.WriteLine($"  {value:G17}");

            var tempered = new uint[5];
            for (var i = 0; i < tempered.Length; i++)
                tempered[i] = generator.NextTemperedUInt32();

            Console.WriteLine();
            Console.WriteLine("Next five tempered 32-bit values:");
            foreach (var value in tempered)
                Console.WriteLine($"  {value}");

            var reseeded = MersenneTwisterWrapper.Create(seed);

            var reseededDoubles = new double[doubles.Length];
            for (var i = 0; i < reseededDoubles.Length; i++)
                reseededDoubles[i] = reseeded.NextDouble();

            var reseededTempered = new uint[tempered.Length];
            for (var i = 0; i < reseededTempered.Length; i++)
                reseededTempered[i] = reseeded.NextTemperedUInt32();

            Console.WriteLine();
            Console.WriteLine($"Reseeding reproduces double sequence: {doubles.SequenceEqual(reseededDoubles)}");
            Console.WriteLine($"Reseeding reproduces tempered sequence: {tempered.SequenceEqual(reseededTempered)}");

            // Expected output:
            //  First five doubles from the generator:
            //    0.6294473727863579
            //    0.8115838741512384
            //    0.2539736325870121
            //    0.8267517122780388
            //    0.264718492450819
            //
            //  Next five tempered 32-bit values:
            //    418932835
            //    2350294565
            //    1196140740
            //    809094426
            //    2348838239
            //
            //  Reseeding reproduces double sequence: True
            //  Reseeding reproduces tempered sequence: True
        }

        private sealed class MersenneTwisterWrapper
        {
            private readonly object _instance;
            private readonly Func<double> _nextDouble;
            private readonly Func<uint> _nextTemperedUInt32;

            private MersenneTwisterWrapper(object instance, Func<double> nextDouble, Func<uint> nextTemperedUInt32)
            {
                _instance = instance;
                _nextDouble = nextDouble;
                _nextTemperedUInt32 = nextTemperedUInt32;
            }

            public static MersenneTwisterWrapper Create(uint seed)
            {
                var coreAssembly = typeof(MLContext).Assembly;
                var type = coreAssembly.GetType("Microsoft.ML.Internal.Utilities.MersenneTwister", throwOnError: true)!;

                var instance = Activator.CreateInstance(type, seed)!;

                var nextDouble = (Func<double>)type
                    .GetMethod(nameof(NextDouble), BindingFlags.Public | BindingFlags.Instance)!
                    .CreateDelegate(typeof(Func<double>), instance);

                var nextTempered = (Func<uint>)type
                    .GetMethod("NextTemperedUInt32", BindingFlags.Public | BindingFlags.Instance, null, Type.EmptyTypes, null)!
                    .CreateDelegate(typeof(Func<uint>), instance);

                return new MersenneTwisterWrapper(instance, nextDouble, nextTempered);
            }

            public double NextDouble() => _nextDouble();

            public uint NextTemperedUInt32() => _nextTemperedUInt32();
        }
    }
}
