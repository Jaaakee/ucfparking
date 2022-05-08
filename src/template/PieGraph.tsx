import { PieChart, Pie, Cell, Legend, Tooltip } from 'recharts';

import { ChartCard } from '../chart/ChartCard';

const data = [
  { name: 'Garage A', value: 344 },
  { name: 'Garage B', value: 657 },
  { name: 'Garage C', value: 345 },
  { name: 'Garage D', value: 200 },
  { name: 'Garage H', value: 456 },
  { name: 'Garage I', value: 323 },
  { name: 'Libra Garage', value: 672 },
];

const PieGraph = () => (
  <ChartCard title="Spaces Filled by Garage">
    <PieChart>
      <Pie data={data} dataKey="value">
        <Cell fill="#A3BFFA" />
        <Cell fill="#7F9CF5" />
        <Cell fill="#667EEA" />
        <Cell fill="#5A67D8" />
      </Pie>
      <Legend />
      <Tooltip />
    </PieChart>
  </ChartCard>
);

export { PieGraph };
