import {
  AreaChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Area,
} from 'recharts';

import { ChartCard } from '../chart/ChartCard';

const data = [
  {
    users: 10,
    month: 'Apr',
  },
  {
    users: 12,
    month: 'May',
  },
  {
    users: 20,
    month: 'Jun',
  },
  {
    users: 25,
    month: 'Jul',
  },
  {
    users: 50,
    month: 'Aug',
  },
  {
    users: 35,
    month: 'Sep',
  },
  {
    users: 42,
    month: 'Oct',
  },
  {
    users: 62,
    month: 'Nov',
  },
  {
    users: 72,
    month: 'Dec',
  },
];

const LineGraph = () => (
  <ChartCard title="Parking Data">
    <AreaChart
      data={data}
      margin={{
        top: 0,
        right: 28,
        left: 0,
        bottom: 0,
      }}
    >
      <XAxis dataKey="month" tickLine={false} axisLine={false} />
      <YAxis tickLine={false} axisLine={false} />
      <CartesianGrid stroke="#E5E7EB" strokeDasharray="15" vertical={false} />
      <Tooltip />
      <Area
        type="monotone"
        dataKey="users"
        name="Users"
        strokeWidth={5}
        stroke="#667EEA"
        fillOpacity={0}
      />
    </AreaChart>
  </ChartCard>
);

export { LineGraph };
