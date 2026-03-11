export function BasketballCourt({ opacity = 0.05 }: { opacity?: number }) {
  return (
    <svg viewBox="0 0 500 940" fill="none" xmlns="http://www.w3.org/2000/svg" style={{ width: '100%', height: '100%', opacity }}>
      {/* Outer boundary */}
      <rect x="10" y="10" width="480" height="920" rx="4" stroke="#ff6b35" strokeWidth="2" fill="none" />
      {/* Half-court line */}
      <line x1="10" y1="470" x2="490" y2="470" stroke="#ff6b35" strokeWidth="1.5" />
      {/* Center circle */}
      <circle cx="250" cy="470" r="60" stroke="#ff6b35" strokeWidth="1.5" fill="none" />
      <circle cx="250" cy="470" r="6" fill="#ff6b35" fillOpacity="0.3" />
      {/* Top paint */}
      <rect x="170" y="10" width="160" height="190" stroke="#ff6b35" strokeWidth="1.5" fill="none" />
      <path d="M 170 200 A 80 80 0 0 0 330 200" stroke="#ff6b35" strokeWidth="1.5" fill="none" />
      <path d="M 186 10 A 60 60 0 0 0 314 10" stroke="#ff6b35" strokeWidth="1" fill="none" strokeDasharray="5,5" />
      {/* Top 3-pt arc */}
      <path d="M 60 10 L 60 150 A 190 190 0 0 0 440 150 L 440 10" stroke="#ff6b35" strokeWidth="1.5" fill="none" />
      {/* Bottom paint (mirrored) */}
      <rect x="170" y="740" width="160" height="190" stroke="#ff6b35" strokeWidth="1.5" fill="none" />
      <path d="M 170 740 A 80 80 0 0 1 330 740" stroke="#ff6b35" strokeWidth="1.5" fill="none" />
      {/* Bottom 3-pt arc */}
      <path d="M 60 930 L 60 790 A 190 190 0 0 1 440 790 L 440 930" stroke="#ff6b35" strokeWidth="1.5" fill="none" />
    </svg>
  );
}
