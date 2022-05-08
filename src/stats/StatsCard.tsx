import { ReactNode } from 'react';

type IStatsCardProps = {
  icon: ReactNode;
  text: string;
  children: ReactNode;
};

const StatsCard = (props: IStatsCardProps) => (
  <div className="stats-card flex items-center border border-gray-200 bg-white rounded-md p-4">
    <div className="w-16 h-16 flex items-center justify-center rounded-full bg-indigo-400 flex-shrink-0">
      {props.icon}
    </div>

    <div className="ml-4">
      <div className="text-xl font-bold text-gray-800">{props.children}</div>
      <div className="text-lg font-semibold">{props.text}</div>
    </div>

    <style jsx>
      {`
        .stats-card :global(svg) {
          @apply text-gray-100 stroke-current w-8 h-8 stroke-2;
        }
      `}
    </style>
  </div>
);

export { StatsCard };
