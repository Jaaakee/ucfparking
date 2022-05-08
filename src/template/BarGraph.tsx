import { BarChart, XAxis, YAxis, CartesianGrid, Tooltip, Bar } from 'recharts';

import { ChartCard } from '../chart/ChartCard';

const data = [
  {
    name: 'Garage A',
    spaces: 4000
  },
  {
    name: 'Garage B',
    spaces: 3000
  },
  {
    name: 'Garage C',
    spaces: 2000
  },
  {
    name: 'Garage D',
    spaces: 2780
  },
  {
    name: 'Garage H',
    spaces: 1890
  },
  {
    name: 'Garage I',
    spaces: 2390
  },
  {
    name: 'Libra Garage',
    spaces: 3490
  },
];

const BarGraph = () => (
  <ChartCard title="Current Spaces Available">
    <BarChart
      data={data}
      margin={{
        top: 0,
        right: 28,
        left: 25,
        bottom: 0,
      }}
    >
      <XAxis dataKey="name" tickLine={false} axisLine={false} />
      <YAxis tickLine={false} axisLine={false} />
      <CartesianGrid stroke="#E5E7EB" strokeDasharray="15" vertical={false} />
      <Tooltip cursor={{ fill: 'rgb(156, 163, 175, 0.2)' }} />
      <Bar dataKey="spaces" name="Page View" fill="#667EEA" fillOpacity={0.6} />
    </BarChart>
  </ChartCard>
);

export { BarGraph };
