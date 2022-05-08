import { PieChart, Pie, Cell, Legend, Tooltip } from 'recharts';

import { ChartCard } from '../chart/ChartCard';

const data = [
  { name: 'Garage A', value: 344 },
  { name: 'Garage B', value: 657 },
  { name: 'Garage C', value: 345 },
  { name: 'Garage D', value: 200 },
  { name: 'Garage H', value: 456 },
  { name: 'Garage I', value: 323 },
  { name: 'Libra Garage', value: 672 }
];

const PieGraph = () => (
  <ChartCard title="Spaces Filled by Garage">
    <PieChart>
      <Pie data={data} dataKey="value">
        <Cell fill="#3694da" />
        <Cell fill="#0d9252" />
        <Cell fill="#556d8c" />
        <Cell fill="#674ea7" />
        <Cell fill="#dc1010" />
        <Cell fill="#b7950c" />
        <Cell fill="#433737" />
      </Pie>
      <Legend />
      <Tooltip />
    </PieChart>
  </ChartCard>
);

export { PieGraph };
