import { useSelector } from 'react-redux'
import {
  BarChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Bar,
  Cell
} from 'recharts'
import { ChartCard } from '../chart/ChartCard'
const data = [
  {
      "name": "Garage A",
      "spaces": 1622
  },
  {
      "name": "Garage B",
      "spaces": 1118
  },
  {
      "name": "Garage C",
      "spaces": 1854
  },
  {
      "name": "Garage D",
      "spaces": 1243
  },
  {
      "name": "Garage H",
      "spaces": 1284
  },
  {
      "name": "Garage I",
      "spaces": 1231
  },
  {
      "name": "Garage Libra",
      "spaces": 1062
  }
]
const colors = [
  '#3694da',
  '#0d9252',
  '#556d8c',
  '#674ea7',
  '#dc1010',
  '#b7950c',
  '#433737'
]



const BarGraph = () => {
  // eslint-disable-next-line @typescript-eslint/no-shadow
  const state = useSelector((state: any) => state.chart);
  return (
    <ChartCard title="Current Spaces Available">
      <BarChart
        data={state.barChartData.length === 0 ? data : state.barChartData}
        barCategoryGap="5%"
        margin={{
          top: 0,
          right: 28,
          left: 25,
          bottom: 0
        }}
      >
        <CartesianGrid className="dark:stroke-slate-600" strokeDasharray="1 1" />
        <XAxis dataKey="name" tickLine={false} axisLine={false} />
        <YAxis tickLine={false} axisLine={false} />
        <CartesianGrid stroke="#E5E7EB" strokeDasharray="15" vertical={false} />
        <Tooltip cursor={{ fill: 'rgb(156, 163, 175, 0.2)' }} />
        {/* <Bar dataKey="spaces" name="Page View" fill="#667EEA" fillOpacity={0.6}> */}
        <Bar dataKey="spaces" name="Available Spaces" fill="#667EEA">
          {(state.barChartData.length === 0 ? data : state.barChartData).map((_entry: any, index: number) => (
            <Cell key={`cell-${index}`} fill={colors[index % 7]} />
          ))}
        </Bar>
      </BarChart>
    </ChartCard>
  )
}

export { BarGraph }
