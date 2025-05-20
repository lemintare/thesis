'use client';
import React, { useState, useEffect } from 'react';

const API_URL = 'http://localhost:8000';

export default function App() {
  const [tableData, setTableData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        const res = await fetch(`${API_URL}/vehicles`);
        if (!res.ok) throw new Error(res.statusText);

        const raw = await res.json();
        const mapped = raw.map((d) => ({
          ...d,
          datetime: new Date(d.datetime ?? d.entry_time).toLocaleString(),
        }));
        setTableData(mapped);
      } catch (err) {
        console.error('[fetchData]', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();                      
    const id = setInterval(fetchData, 30_000);
    return () => clearInterval(id); 
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-6">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-center">Стрим и данные</h1>
        <p className="text-gray-400 mt-2 text-center">Таблица обновляется каждые 30&nbsp;секунд</p>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-black rounded-lg overflow-hidden shadow-xl border border-gray-700">
          <div className="aspect-video relative">
            <img
              src="http://localhost:8000/stream-feed"
              alt="Стрим"
              className="w-full h-full object-cover"
            />
            <div className="absolute top-4 left-4 bg-red-600 text-white px-3 py-1 rounded-full text-sm font-semibold">
              LIVE
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg shadow-xl border border-gray-700 p-4 flex flex-col">
          <h2 className="text-xl font-semibold mb-4">Последние записи</h2>

          <div className="flex-1 overflow-auto">
            {isLoading ? (
              <div className="animate-pulse space-y-3">
                {Array.from({ length: 10 }, (_, i) => (
                  <div key={i} className="h-10 bg-gray-700 rounded" />
                ))}
              </div>
            ) : tableData.length === 0 ? (
              <div className="text-gray-400 text-center py-4">
                Нет записей
              </div>
            ) : (
              <table className="w-full table-auto">
                <thead>
                  <tr className="text-left text-gray-400 text-sm">
                    <th className="pb-3">Номер</th>
                    <th className="pb-3">Дата и&nbsp;время</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-700">
                  {tableData.map((row) => (
                    <tr key={row.id} className="hover:bg-gray-700 transition-colors">
                      <td className="py-2 font-mono">{row.plate_number}</td>
                      <td className="py-2 text-sm">{row.datetime}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </main>

      <footer className="mt-12 text-center text-gray-500 text-sm">
        <p>© 2025 Система мониторинга. Все права защищены.</p>
      </footer>
    </div>
  );
}
