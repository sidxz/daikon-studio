import type { SVGProps } from "react";

/**
 * Navigation glyphs on lucide's 24-unit grid and 2-unit stroke, so they sit
 * beside the lucide icons that remain on buttons. One vocabulary: a filled
 * dot is a measured value, an open ring is a prediction, as in the logo.
 */
function Glyph({ children, ...props }: SVGProps<SVGSVGElement>) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  );
}
const Dot = (p: { cx: number; cy: number; r: number }) => (
  <circle {...p} fill="currentColor" stroke="none" />
);

export const DatasetsIcon = (props: SVGProps<SVGSVGElement>) => (
  <Glyph {...props}>
    <rect x="3.5" y="4" width="17" height="16" rx="2" />
    <path d="M3.5 9.5h17M3.5 14.5h17M11 12h6M11 17.2h4" />
    <Dot cx={7.4} cy={12} r={1.35} />
    <Dot cx={7.4} cy={17.2} r={1.35} />
  </Glyph>
);
export const ProtocolsIcon = (props: SVGProps<SVGSVGElement>) => (
  <Glyph {...props}>
    <path d="M12 2.8 19.9 7.4v9.2L12 21.2l-7.9-4.6V7.4z" />
    <path d="m8.6 12.2 2.4 2.4 4.5-4.8" />
  </Glyph>
);
export const SweepsIcon = (props: SVGProps<SVGSVGElement>) => (
  <Glyph {...props}>
    <Dot cx={4.5} cy={12} r={1.9} />
    <path d="M6.3 11.1 17 5.6M6.6 12h10.2M6.3 12.9 17 18.4" />
    <circle cx="19" cy="4.6" r="2" />
    <circle cx="19" cy="12" r="2" />
    <circle cx="19" cy="19.4" r="2" />
  </Glyph>
);
export const RunsIcon = (props: SVGProps<SVGSVGElement>) => (
  <Glyph {...props}>
    <path d="M4.5 5v14l9-7z" />
    <circle cx="19" cy="8" r="2.2" />
    <circle cx="19" cy="16" r="2.2" />
  </Glyph>
);
export const CollectionsIcon = (props: SVGProps<SVGSVGElement>) => (
  <Glyph {...props}>
    <rect x="3" y="3" width="18" height="18" rx="4" strokeDasharray="3.2 2.6" />
    <Dot cx={9} cy={9.5} r={1.5} />
    <Dot cx={15} cy={10.2} r={1.5} />
    <Dot cx={11.4} cy={15} r={1.5} />
  </Glyph>
);
export const EnginesIcon = (props: SVGProps<SVGSVGElement>) => (
  <Glyph {...props}>
    <path d="M3.5 3.5v17h17" />
    <path d="M7 17c4.5 0 6.5-10 13-10.5" />
    <Dot cx={9.6} cy={13.4} r={1.4} />
    <Dot cx={14.8} cy={13.4} r={1.4} />
    <Dot cx={17.6} cy={3.8} r={1.4} />
  </Glyph>
);
export const RunnersIcon = (props: SVGProps<SVGSVGElement>) => (
  <Glyph {...props}>
    <rect x="3" y="4" width="18" height="7" rx="2" />
    <rect x="3" y="13" width="18" height="7" rx="2" />
    <Dot cx={7} cy={7.5} r={1.3} />
    <Dot cx={7} cy={16.5} r={1.3} />
    <path d="M11 7.5h6M11 16.5h3" />
  </Glyph>
);
