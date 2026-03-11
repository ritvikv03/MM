// Epanechnikov kernel density estimation

export function epanechnikovKernel(bandwidth: number) {
  return (x: number, xi: number): number => {
    const u = (x - xi) / bandwidth;
    return Math.abs(u) <= 1 ? (0.75 * (1 - u * u)) / bandwidth : 0;
  };
}

export function silvermanBandwidth(data: number[]): number {
  const n = data.length;
  if (n <= 1) return 1;
  const mean = data.reduce((a, b) => a + b, 0) / n;
  const variance = data.reduce((acc, v) => acc + (v - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const iqr = interquartileRange(data);
  const s = Math.min(std, iqr / 1.34);
  const bw = 0.9 * s * Math.pow(n, -0.2);
  return Math.max(bw, 0.01); // minimum bandwidth floor to handle identical samples
}

export function interquartileRange(data: number[]): number {
  const sorted = [...data].sort((a, b) => a - b);
  const n = sorted.length;
  if (n === 0) return 1;
  const q1 = sorted[Math.floor(n * 0.25)];
  const q3 = sorted[Math.floor(n * 0.75)];
  return q3 - q1 || 1;
}

export function linspace(start: number, stop: number, num: number): number[] {
  if (num <= 0) return [];
  if (num === 1) return [start];
  const step = (stop - start) / (num - 1);
  return Array.from({ length: num }, (_, i) => start + i * step);
}

export function computeKDE(
  data: number[],
  gridPoints: number[],
  bandwidth?: number,
): number[] {
  const bw = bandwidth ?? silvermanBandwidth(data);
  const kernel = epanechnikovKernel(bw);
  return gridPoints.map((x) =>
    data.reduce((sum, xi) => sum + kernel(x, xi), 0) / data.length,
  );
}

export function kdeIntersectionArea(
  density1: number[],
  density2: number[],
  gridPoints: number[],
): number {
  const n = gridPoints.length;
  if (n < 2) return 0;
  let area = 0;
  for (let i = 0; i < n - 1; i++) {
    const minDensity = Math.min(density1[i], density2[i]);
    const dx = gridPoints[i + 1] - gridPoints[i];
    area += minDensity * dx;
  }
  return area;
}
