import { useSelector } from 'react-redux'
import { PieChart, Pie, Cell, Legend, Tooltip } from 'recharts'

import { ChartCard } from '../chart/ChartCard'

const data = [
  {
      "name": "Garage A",
      "value": 1622
  },
  {
      "name": "Garage B",
      "value": 1118
  },
  {
      "name": "Garage C",
      "value": 1854
  },
  {
      "name": "Garage D",
      "value": 1243
  },
  {
      "name": "Garage H",
      "value": 1284
  },
  {
      "name": "Garage I",
      "value": 1231
  },
  {
      "name": "Garage Libra",
      "value": 1062
  }
]

const PieGraph = () => {
  // eslint-disable-next-line @typescript-eslint/no-shadow
  const state = useSelector((state: any) => state.chart);
  return (
    <ChartCard title="Current Spaces Filled by Garage">
      <PieChart>
        <Pie data={state.pieChartData.length === 0 ? data : state.pieChartData} dataKey="value">
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
  )
}

export { PieGraph }
